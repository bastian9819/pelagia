import { initGpu } from './device.js';
import common from './shaders/life_common.wgsl?raw';
import gridPasses from './shaders/life_grid.wgsl?raw';
import simPass from './shaders/life_sim.wgsl?raw';
import cyclePasses from './shaders/life_cycle.wgsl?raw';
import creatureShader from './shaders/render_quad.wgsl?raw';
import foodShader from './shaders/render_food.wgsl?raw';
import fadeShader from './shaders/fade.wgsl?raw';
import presentShader from './shaders/present.wgsl?raw';
import { DEFAULT_CONFIG } from '../core/config.js';
import { GENOME_SIZE, WEIGHT_GENES, SIZE_GENE } from '../sim/brain.js';
import { pcgHash, floatFromU32, Rng } from '../core/rng.js';
import { SpatialGrid } from '../sim/grid.js';
import { wrapDelta } from '../core/space.js';
import { buildUi, mkBtn } from './ui.js';
import { buildBrainView } from './brainView.js';
import {
  buildLineagePanel,
  characterizeGenome,
  type LineageRow,
  type LineageTraits,
} from './lineages.js';
import { buildGodPanel, type GodSpec } from './god.js';
import {
  buildObservatory,
  buildEvolutionHistory,
  type ObservatoryData,
  type WorldSample,
  type LineageHistory,
  type Watched,
  type WatchSample,
} from './observatory.js';
import { t, onLang } from './i18n.js';
import { parseShareState, buildShareUrl, applyHash, copyToClipboard } from './share.js';
import inspectShader from './shaders/inspect.wgsl?raw';

/**
 * Phase 2.0: the full evolving ecosystem on the GPU — grid build, sense, brain,
 * move, eat (atomic claim), metabolise, death (free-list) and energy-threshold
 * reproduction into freed slots, plus food respawn. Fixed-capacity buffer (N
 * slots); alive count floats as creatures die and are reborn.
 *
 * Exposes window.__pelagia.readChemotaxis() so emergence can be validated on the
 * GPU the same way as the CPU oracle (mean cos(heading -> nearest food)).
 */
export interface OceanOptions {
  /** Requested creature slot count (capped by GPU memory). */
  n: number;
  /** Sim ticks to run headless before the first rendered frame. */
  warmup?: number;
  /** Called once the ocean is set up and warmed (to dismiss a loading screen). */
  onReady?: () => void;
}

export async function runGpuSim(canvas: HTMLCanvasElement, opts: OceanOptions): Promise<void> {
  // Phase 5: a shared link restores seed + N + god-mode params; a fresh visit
  // gets a random seed so every ocean is novel (and worth sharing). The seed
  // drives both the CPU-side init below and the GPU shader RNG (via pu[23]).
  const share = parseShareState();
  const seed = (share?.seed ?? Math.floor(Math.random() * 0x100000000)) >>> 0;
  const requestedN = share?.n ?? opts.n;
  const { device, format } = await initGpu();
  device.addEventListener('uncapturederror', (e) => {
    // Surface async WebGPU validation errors (otherwise the whole command
    // buffer is silently dropped -> black screen).
    console.error('WebGPU error:', e.error.message);
  });
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('webgpu canvas context unavailable');
  context.configure({ device, format, alphaMode: 'opaque' });

  const cfg = DEFAULT_CONFIG;
  const cellSize = cfg.perceptionRadius;
  const maxN = Math.floor(device.limits.maxStorageBufferBindingSize / (GENOME_SIZE * 4));
  const n = Math.min(requestedN, maxN);
  const f = n; // food slot pool
  // Food must be SCARCE relative to creatures so poor foragers starve -> slots
  // free up -> good foragers reproduce -> selection. Only a fraction start alive.
  const foodInitAlive = Math.round(n * 0.4);
  const world = Math.max(cellSize * 3, 50 * Math.sqrt(n));
  const cols = Math.ceil(world / cellSize);
  const rows = Math.ceil(world / cellSize);
  const numCells = cols * rows;

  // --- Initial data (seeded: same seed -> same starting ocean) ---
  const stateData = new Float32Array(n * 4);
  const bioData = new Float32Array(n * 4);
  const weightsData = new Float32Array(n * GENOME_SIZE);
  const foodData = new Float32Array(f * 2);
  // Independent RNG streams so positions, genomes and food don't couple, and
  // capping N (low-memory device) still gives the first n creatures the same
  // draws as a higher-N run of the same seed (graceful degradation, R-002).
  // Re-runnable so "restart" can reseed the same ocean in place (no page reload).
  function fillInitialData(): void {
    const rngState = new Rng(seed, 1);
    const rngWeights = new Rng(seed, 2);
    const rngFood = new Rng(seed, 3);
    for (let i = 0; i < n; i++) {
      stateData[i * 4 + 0] = rngState.nextFloat() * world;
      stateData[i * 4 + 1] = rngState.nextFloat() * world;
      stateData[i * 4 + 2] = rngState.nextFloat() * Math.PI * 2;
      stateData[i * 4 + 3] = 0; // speed (reset on restart)
      bioData[i * 4 + 0] = cfg.initialEnergy;
      bioData[i * 4 + 1] = floatFromU32(pcgHash(i)); // hue derived from lineage id
      bioData[i * 4 + 2] = 1; // alive
      bioData[i * 4 + 3] = i; // lineage id (= founder slot; inherited unchanged)
    }
    for (let i = 0; i < n; i++) {
      const wb = i * GENOME_SIZE;
      for (let k = 0; k < WEIGHT_GENES; k++) {
        weightsData[wb + k] = (rngWeights.nextFloat() * 2 - 1) * cfg.weightInitStd;
      }
      // Activation genes biased active (uniform(-0.3, 1.0), ~77% on) so brains
      // start capable; mutation then evolves how many neurons each lineage uses.
      for (let k = WEIGHT_GENES; k < SIZE_GENE; k++) {
        weightsData[wb + k] = rngWeights.nextFloat() * 1.3 - 0.3;
      }
      // Body-size gene with real spread (~N(1, 0.4) bodies) so predator/prey size
      // niches can form from the start instead of waiting for variance to build.
      weightsData[wb + SIZE_GENE] = rngWeights.nextGaussian() * 0.8;
    }
    for (let j = 0; j < f; j++) {
      if (j < foodInitAlive) {
        foodData[j * 2 + 0] = rngFood.nextFloat() * world;
        foodData[j * 2 + 1] = rngFood.nextFloat() * world;
      } else {
        foodData[j * 2 + 0] = -1; // dead slot (sentinel)
        foodData[j * 2 + 1] = -1;
      }
    }
  }
  fillInitialData();

  // --- Buffers ---
  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const CS = GPUBufferUsage.COPY_SRC;
  const stateBuf = device.createBuffer({ size: stateData.byteLength, usage: S | CD | CS });
  const bioBuf = device.createBuffer({ size: bioData.byteLength, usage: S | CD | CS });
  const weightsBuf = device.createBuffer({ size: weightsData.byteLength, usage: S | CD | CS });
  const foodBuf = device.createBuffer({ size: foodData.byteLength, usage: S | CD | CS });
  // Phase 6: gridData / cellStart / sortedIdx hold BOTH the food grid and a
  // second CREATURE grid (neighbour perception + predation), packed into the same
  // buffers to stay under the 8-storage-buffer limit (see life_common.wgsl).
  // gridData also carries freeCount, food budget, the predation counter, food
  // claims and per-creature predation claims. COPY_DST so restart can reset
  // freeCount in place.
  const gridDataBuf = device.createBuffer({
    size: (2 * numCells + 3 + f + n) * 4,
    usage: S | CS | CD,
  });
  const cellStartBuf = device.createBuffer({ size: 2 * (numCells + 1) * 4, usage: S });
  const sortedBuf = device.createBuffer({ size: (f + n) * 4, usage: S });
  const freeListBuf = device.createBuffer({ size: n * 4, usage: S });
  // Brain-inspector output (one selected creature): inputs|hidden|outputs|state.
  const INSPECT_FLOATS = 32;
  const inspectBuf = device.createBuffer({ size: INSPECT_FLOATS * 4, usage: S | CS });
  const inspectReadback = device.createBuffer({
    size: INSPECT_FLOATS * 4,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  // Watch-list read-back: a few tracked creatures sampled through the inspect pass.
  const MAX_WATCH = 8;
  const watchReadback = device.createBuffer({
    size: MAX_WATCH * INSPECT_FLOATS * 4,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  device.queue.writeBuffer(stateBuf, 0, stateData);
  device.queue.writeBuffer(bioBuf, 0, bioData);
  device.queue.writeBuffer(weightsBuf, 0, weightsData);
  device.queue.writeBuffer(foodBuf, 0, foodData);

  // --- Params uniform (112 bytes: 5 vec4<f32> + 2 vec4<u32>) ---
  // Phase 6 appended a 5th f32 vec4 `ext` AFTER d0/d1, so existing indices are
  // unchanged: d0 = pu[16..19], d1 = pu[20..23], ext = pf[24..27].
  const paramsBuf = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | CD });
  const pbuf = new ArrayBuffer(112);
  const pf = new Float32Array(pbuf);
  const pu = new Uint32Array(pbuf);
  pf[0] = world;
  pf[1] = world;
  pf[2] = cellSize;
  pf[3] = cfg.maxSpeed;
  pf[4] = cfg.maxTurnRate;
  pf[5] = 1; // dt
  pf[6] = cfg.eatRadius;
  pf[7] = cfg.foodEnergy;
  pf[8] = cfg.baseCost;
  pf[9] = cfg.moveCost;
  pf[10] = cfg.reproductionThreshold;
  pf[11] = cfg.initialEnergy;
  pf[12] = cfg.mutationRate;
  pf[13] = cfg.mutationStd;
  pf[14] = cfg.offspringEnergyFraction;
  // Food spawn budget per tick. Caps energy influx so carrying capacity sits
  // well below N -> heavy death/birth turnover -> strong selection. Matched to
  // the CPU ecology's food-influx density; calibrated via alive + chemotaxis.
  pf[15] = Math.max(1, Math.round(n * 0.005));
  pu[16] = cols;
  pu[17] = rows;
  pu[18] = numCells;
  pu[19] = n;
  pu[20] = f;
  pu[23] = seed >>> 0; // world seed: folded into the shader RNG (mutation/respawn)
  // Predation gain (fraction of prey energy the predator gains; 0 disables it) and
  // predation margin (how much bigger you must be to eat another, >= 1). Both are
  // god-mode sliders, serialised in the share URL. Moderate defaults = food stays
  // the primary resource, predation a secondary pressure (anti-gray-soup, R-001).
  pf[24] = 0.5;
  pf[25] = 1.25;
  // Food patchiness (ext.z): 0 = uniform sprinkle, 1 = tight drifting blooms.
  pf[26] = 0.6;
  // Big-food value (ext.w): energy multiplier of a rare big-food pellet vs plankton.
  pf[27] = 5;
  function writeParams(frame: number): void {
    pu[21] = frame;
    device.queue.writeBuffer(paramsBuf, 0, pbuf);
  }

  // --- Bind groups (two, to stay under the per-stage storage limit) ---
  const u = GPUShaderStage.COMPUTE;
  const sto: GPUBufferBindingLayout = { type: 'storage' };
  const bgl0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: u, buffer: { type: 'uniform' } },
      { binding: 1, visibility: u, buffer: sto },
      { binding: 2, visibility: u, buffer: sto },
      { binding: 3, visibility: u, buffer: sto },
      { binding: 4, visibility: u, buffer: sto },
    ],
  });
  const bgl1 = device.createBindGroupLayout({
    entries: [0, 1, 2, 3].map((binding) => ({ binding, visibility: u, buffer: sto })),
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
  const group0 = device.createBindGroup({
    layout: bgl0,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } },
      { binding: 4, resource: { buffer: foodBuf } },
    ],
  });
  const group1 = device.createBindGroup({
    layout: bgl1,
    entries: [
      { binding: 0, resource: { buffer: gridDataBuf } },
      { binding: 1, resource: { buffer: cellStartBuf } },
      { binding: 2, resource: { buffer: sortedBuf } },
      { binding: 3, resource: { buffer: freeListBuf } },
    ],
  });

  const gridModule = device.createShaderModule({ code: common + '\n' + gridPasses });
  const simModule = device.createShaderModule({ code: common + '\n' + simPass });
  const cycleModule = device.createShaderModule({ code: common + '\n' + cyclePasses });
  const wgslErrors: string[] = [];
  for (const [name, mod] of [
    ['grid', gridModule],
    ['sim', simModule],
    ['cycle', cycleModule],
  ] as const) {
    const info = await mod.getCompilationInfo();
    for (const m of info.messages) {
      if (m.type === 'error')
        wgslErrors.push(`[WGSL ${name}] ${m.lineNum}:${m.linePos} ${m.message}`);
    }
  }
  if (wgslErrors.length) throw new Error('Shader compile errors:\n' + wgslErrors.join('\n'));
  const mk = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
    device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
  const pGridClear = mk(gridModule, 'gridClear');
  const pClaimClear = mk(gridModule, 'claimClear');
  const pCount = mk(gridModule, 'count');
  const pCountCreatures = mk(gridModule, 'countCreatures');
  const pScan = mk(gridModule, 'scan');
  const pScatter = mk(gridModule, 'scatter');
  const pScatterCreatures = mk(gridModule, 'scatterCreatures');
  const pSim = mk(simModule, 'main');
  const pDeath = mk(cycleModule, 'death');
  const pRepro = mk(cycleModule, 'repro');
  const pRespawn = mk(cycleModule, 'foodRespawn');

  // Brain inspector: its own bind group (uniform + 6 read-only + 1 storage = 7).
  const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
  const bglInspect = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: u, buffer: { type: 'uniform' } },
      { binding: 1, visibility: u, buffer: ro },
      { binding: 2, visibility: u, buffer: ro },
      { binding: 3, visibility: u, buffer: ro },
      { binding: 4, visibility: u, buffer: ro },
      { binding: 5, visibility: u, buffer: ro },
      { binding: 6, visibility: u, buffer: ro },
      { binding: 7, visibility: u, buffer: sto },
    ],
  });
  const inspectPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglInspect] }),
    compute: { module: device.createShaderModule({ code: inspectShader }), entryPoint: 'main' },
  });
  const inspectBind = device.createBindGroup({
    layout: bglInspect,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } },
      { binding: 4, resource: { buffer: foodBuf } },
      { binding: 5, resource: { buffer: cellStartBuf } },
      { binding: 6, resource: { buffer: sortedBuf } },
      { binding: 7, resource: { buffer: inspectBuf } },
    ],
  });

  const wgN = Math.ceil(n / 256);
  const wgF = Math.ceil(f / 256);
  const wgC2 = Math.ceil((2 * numCells) / 256); // clear both grids' cell counts
  function recordSim(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, group0);
    pass.setBindGroup(1, group1);
    pass.setPipeline(pGridClear);
    pass.dispatchWorkgroups(wgC2);
    pass.setPipeline(pClaimClear);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pCount);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pCountCreatures);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pScan);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pScatter);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pScatterCreatures);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pSim);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pDeath);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pRepro);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pRespawn);
    pass.dispatchWorkgroups(wgF);
    pass.end();
  }

  // --- Render: HDR accumulation texture -> trails + glow -> tonemapped present ---
  const renderUbo = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | CD });
  const renderData = new Float32Array(8);
  renderData[7] = Math.floor(f / 16); // big-food slot count for the food renderer
  const ACCUM_FORMAT: GPUTextureFormat = 'rgba16float';
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const additive: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  };
  const overBlend: GPUBlendState = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };
  const mkRender = (
    code: string,
    blend: GPUBlendState | undefined,
    target: GPUTextureFormat,
  ): GPURenderPipeline => {
    const mod = device.createShaderModule({ code });
    const colorTarget: GPUColorTargetState = blend ? { format: target, blend } : { format: target };
    return device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [colorTarget] },
      primitive: { topology: 'triangle-list' },
    });
  };
  const creaturePipeline = mkRender(creatureShader, additive, ACCUM_FORMAT);
  const foodPipeline = mkRender(foodShader, additive, ACCUM_FORMAT);
  const fadePipeline = mkRender(fadeShader, overBlend, ACCUM_FORMAT);
  const presentPipeline = mkRender(presentShader, undefined, format);

  const creatureBind = device.createBindGroup({
    layout: creaturePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUbo } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } }, // body-size gene for sizing
    ],
  });
  const foodBind = device.createBindGroup({
    layout: foodPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUbo } },
      { binding: 1, resource: { buffer: foodBuf } },
    ],
  });

  let accumTex: GPUTexture | undefined;
  let accumView!: GPUTextureView;
  let presentBind!: GPUBindGroup;
  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.floor(window.innerWidth * dpr));
    const ch = Math.max(1, Math.floor(window.innerHeight * dpr));
    canvas.width = cw;
    canvas.height = ch;

    accumTex?.destroy();
    accumTex = device.createTexture({
      size: [cw, ch],
      format: ACCUM_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    accumView = accumTex.createView();
    presentBind = device.createBindGroup({
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: accumView },
      ],
    });
    // Clear the fresh accumulation texture once.
    const enc = device.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: accumView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      .end();
    device.queue.submit([enc.finish()]);
    updateView();
  }

  // --- Camera (zoom + pan) ---
  const cam = { zoom: 1, cx: world / 2, cy: world / 2 };
  function ppwNow(): number {
    return Math.min(canvas.width / world, canvas.height / world) * cam.zoom;
  }
  function clampCam(): void {
    cam.zoom = Math.min(60, Math.max(1, cam.zoom));
    cam.cx = Math.min(world, Math.max(0, cam.cx));
    cam.cy = Math.min(world, Math.max(0, cam.cy));
  }
  function updateView(): void {
    const cw = canvas.width;
    const ch = canvas.height;
    const ppw = ppwNow();
    renderData[0] = (2 * ppw) / cw; // sx (world -> NDC)
    renderData[2] = (-2 * cam.cx * ppw) / cw; // ox
    renderData[1] = (-2 * ppw) / ch; // sy (flip y)
    renderData[3] = (2 * cam.cy * ppw) / ch; // oy
    // Glow radius in world units, set so the ON-SCREEN size stays in a sane band
    // (~2.5px when zoomed out so creatures are still visible, capped ~16px when
    // zoomed in so they read as creatures, not giant blobs). 5 = creature scale.
    renderData[4] = Math.min(16 / ppw, Math.max(2.5 / ppw, 5));
    renderData[5] = 1.4; // brightness (HDR; tonemapped on present)
    device.queue.writeBuffer(renderUbo, 0, renderData);
  }
  function screenToWorld(clientX: number, clientY: number): { wx: number; wy: number } {
    const dpr = canvas.width / window.innerWidth;
    const ppw = ppwNow();
    return {
      wx: (clientX * dpr - canvas.width / 2) / ppw + cam.cx,
      wy: (clientY * dpr - canvas.height / 2) / ppw + cam.cy,
    };
  }
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const before = screenToWorld(e.clientX, e.clientY);
      cam.zoom *= Math.exp(-e.deltaY * 0.0015);
      clampCam();
      const after = screenToWorld(e.clientX, e.clientY);
      cam.cx += before.wx - after.wx; // keep the point under the cursor fixed
      cam.cy += before.wy - after.wy;
      clampCam();
      updateView();
    },
    { passive: false },
  );
  let dragging = false;
  let dragMoved = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    dragMoved = false;
    lastX = e.clientX;
    lastY = e.clientY;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      /* ignore (e.g. synthetic events) */
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY) > 4) dragMoved = true;
    const ppw = ppwNow();
    const dpr = canvas.width / window.innerWidth;
    cam.cx -= ((e.clientX - lastX) * dpr) / ppw;
    cam.cy -= ((e.clientY - lastY) * dpr) / ppw;
    lastX = e.clientX;
    lastY = e.clientY;
    clampCam();
    updateView();
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    if (!dragMoved) {
      const { wx, wy } = screenToWorld(e.clientX, e.clientY);
      void selectNear(wx, wy);
    }
  });

  resize();
  window.addEventListener('resize', resize);

  // --- Colour-by-trait: how creatures are tinted (renderData[6]); button lives
  // on the transport bar and is driven through this control. ---
  const COLOR_MODES = ['lineageWord', 'sizeWord', 'neurons', 'energyWord', 'speedWord'];
  let colorMode = 0;

  // --- Stats panel + controls ---
  const ui = buildUi(
    () => {
      cam.zoom = 1;
      cam.cx = world / 2;
      cam.cy = world / 2;
      updateView();
    },
    () => {
      // Step one tick (used while paused) so a single decision can be studied.
      writeParams(frame);
      const enc = device.createCommandEncoder();
      recordSim(enc);
      device.queue.submit([enc.finish()]);
      frame++;
    },
    {
      cycle: () => {
        colorMode = (colorMode + 1) % COLOR_MODES.length;
        renderData[6] = colorMode;
        device.queue.writeBuffer(renderUbo, 0, renderData);
      },
      label: () => '🎨 ' + t(COLOR_MODES[colorMode]!),
    },
  );
  document.body.appendChild(ui.panel);
  document.body.appendChild(ui.controls);
  document.body.appendChild(ui.menu);

  // --- Selection + brain inspector ---
  let selectedIndex = -1;
  let selectedLineage = -1; // to detect slot reuse (a different creature took the slot)
  let selecting = false;
  let inspectPending = false;
  const brainView = buildBrainView(
    () => deselect(),
    () => {
      if (selectedIndex >= 0) addWatch(selectedIndex, selectedLineage);
    },
  );
  document.body.appendChild(brainView.panel);
  const ring = document.createElement('div');
  ring.style.cssText =
    'position:fixed;display:none;border:2px solid rgba(63,240,216,0.9);border-radius:50%;' +
    'box-shadow:0 0 12px rgba(63,240,216,0.6);pointer-events:none;transform:translate(-50%,-50%);z-index:5;';
  document.body.appendChild(ring);

  function setSelectedUniform(): void {
    pu[22] = selectedIndex >= 0 ? selectedIndex : 0xffffffff;
  }
  function deselect(): void {
    selectedIndex = -1;
    selectedLineage = -1;
    setSelectedUniform();
    brainView.hide();
    brainView.setGenome(null);
    ring.style.display = 'none';
  }
  // Read one creature's genome back (static per creature) for the policy view.
  async function loadGenome(index: number, lineage: number): Promise<void> {
    const gRead = device.createBuffer({
      size: GENOME_SIZE * 4,
      usage: GPUBufferUsage.MAP_READ | CD,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(weightsBuf, index * GENOME_SIZE * 4, gRead, 0, GENOME_SIZE * 4);
    device.queue.submit([enc.finish()]);
    await gRead.mapAsync(GPUMapMode.READ);
    const g = new Float32Array(gRead.getMappedRange().slice(0));
    gRead.unmap();
    gRead.destroy();
    if (selectedIndex === index && selectedLineage === lineage) brainView.setGenome(g);
  }
  function positionRing(d: Float32Array): void {
    const ppw = ppwNow();
    const dpr = canvas.width / window.innerWidth;
    ring.style.left = `${((d[20]! - cam.cx) * ppw + canvas.width / 2) / dpr}px`;
    ring.style.top = `${((d[21]! - cam.cy) * ppw + canvas.height / 2) / dpr}px`;
    const px = Math.max(18, (renderData[4]! * ppw * 2) / dpr + 10);
    ring.style.width = `${px}px`;
    ring.style.height = `${px}px`;
    ring.style.display = 'block';
  }
  async function selectNear(wx: number, wy: number): Promise<void> {
    if (selecting) return;
    selecting = true;
    const sRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(stateBuf, 0, sRead, 0, n * 16);
    enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
    device.queue.submit([enc.finish()]);
    await Promise.all([sRead.mapAsync(GPUMapMode.READ), bRead.mapAsync(GPUMapMode.READ)]);
    const st = new Float32Array(sRead.getMappedRange().slice(0));
    const bi = new Float32Array(bRead.getMappedRange().slice(0));
    sRead.unmap();
    bRead.unmap();
    sRead.destroy();
    bRead.destroy();

    const maxDist = 26 / ppwNow();
    let best = -1;
    let bestD2 = maxDist * maxDist;
    for (let i = 0; i < n; i++) {
      if (bi[i * 4 + 2]! < 0.5) continue;
      const dx = st[i * 4]! - wx;
      const dy = st[i * 4 + 1]! - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = i;
      }
    }
    selecting = false;
    if (best >= 0) {
      selectedIndex = best;
      selectedLineage = Math.round(bi[best * 4 + 3]!);
      setSelectedUniform();
      brainView.show();
      void loadGenome(best, selectedLineage);
    } else {
      deselect();
    }
  }
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') deselect();
  });

  // --- Lineage explorer ---
  const lineagePanel = buildLineagePanel();
  document.body.appendChild(lineagePanel.panel);
  ui.addTool(lineagePanel.toggle);

  // --- God mode: live world parameters (write straight into the params uniform) ---
  const godDefs: Omit<GodSpec, 'value'>[] = [
    { labelKey: 'g_food', idx: 15, min: 0, max: Math.max(8, n * 0.02), step: 1 },
    { labelKey: 'g_mutRate', idx: 12, min: 0, max: 0.5, step: 0.01 },
    { labelKey: 'g_mutSize', idx: 13, min: 0, max: 1, step: 0.02 },
    { labelKey: 'g_speed', idx: 3, min: 0.5, max: 10, step: 0.1 },
    { labelKey: 'g_agility', idx: 4, min: 0.05, max: 1.2, step: 0.01 },
    { labelKey: 'g_metabolism', idx: 8, min: 0, max: 0.6, step: 0.01 },
    { labelKey: 'g_foodEnergy', idx: 7, min: 2, max: 30, step: 0.5 },
    { labelKey: 'g_reproAt', idx: 10, min: 30, max: 200, step: 1 },
    { labelKey: 'g_offspring', idx: 14, min: 0.2, max: 0.8, step: 0.05 },
    { labelKey: 'g_moveCost', idx: 9, min: 0, max: 0.3, step: 0.01 },
    { labelKey: 'g_predation', idx: 24, min: 0, max: 1, step: 0.05 },
    { labelKey: 'g_predMargin', idx: 25, min: 1, max: 2.5, step: 0.05 },
    { labelKey: 'g_patchiness', idx: 26, min: 0, max: 1, step: 0.05 },
    { labelKey: 'g_bigFood', idx: 27, min: 1, max: 12, step: 0.5 },
  ];
  // Restore shared god params into the uniform BEFORE warmup and before building
  // the sliders, so the warmed-up ocean and the slider positions both reflect the
  // link. Only known god indices are honoured, each value clamped to its slider
  // range (defends against a malformed/hostile hash).
  if (share?.params) {
    for (const { idx, value } of share.params) {
      const def = godDefs.find((d) => d.idx === idx);
      if (def) pf[idx] = Math.min(def.max, Math.max(def.min, value));
    }
    device.queue.writeBuffer(paramsBuf, 0, pbuf);
  }
  const godSpecs: GodSpec[] = godDefs.map((d) => ({ ...d, value: pf[d.idx]! }));
  const godPanel = buildGodPanel(godSpecs, (idx, value) => {
    pf[idx] = value;
    device.queue.writeBuffer(paramsBuf, 0, pbuf); // apply immediately (even if paused)
  });
  document.body.appendChild(godPanel.panel);
  ui.addTool(godPanel.toggle);

  // --- Share: copy a reproducible-ocean URL (seed + N + live god params) ---
  let shareFeedbackTimer = 0;
  const shareBtn = mkBtn('', () => void onShare());
  function relabelShare(): void {
    shareBtn.textContent = '🔗 ' + t('share');
  }
  async function onShare(): Promise<void> {
    const state = { seed, n, params: godPanel.getValues() };
    applyHash(state); // reflect the current ocean in the address bar
    const ok = await copyToClipboard(buildShareUrl(state));
    window.clearTimeout(shareFeedbackTimer);
    shareBtn.textContent = ok ? '✓ ' + t('copied') : '🔗 ' + t('share');
    shareFeedbackTimer = window.setTimeout(relabelShare, 1500);
  }
  relabelShare();
  onLang(relabelShare);
  ui.addTool(shareBtn);

  // --- Restart: reseed the ocean in place (same seed + current params, no reload) ---
  const resetBtn = mkBtn('', () => resetOcean());
  function relabelReset(): void {
    resetBtn.textContent = '↻ ' + t('restart');
  }
  relabelReset();
  onLang(relabelReset);
  ui.addTool(resetBtn);

  // --- Help / pedagogical panel ---
  const help = document.createElement('div');
  help.style.cssText =
    'position:fixed;inset:0;display:none;place-items:center;z-index:20;background:rgba(2,4,10,0.65);';
  const helpCard = document.createElement('div');
  helpCard.style.cssText =
    'max-width:520px;margin:16px;padding:24px 26px;background:rgba(6,18,41,0.97);' +
    'border:1px solid rgba(63,240,216,0.3);border-radius:14px;color:#cfe8ff;' +
    'font:14px/1.65 ui-sans-serif,system-ui,sans-serif;';
  const helpTitle = document.createElement('div');
  helpTitle.style.cssText = 'font-size:20px;font-weight:600;color:#3ff0d8;margin-bottom:12px;';
  const helpBody = document.createElement('div');
  const helpClose = document.createElement('button');
  helpClose.textContent = '✕';
  helpClose.style.cssText =
    'margin-top:16px;padding:8px 16px;background:rgba(63,240,216,0.15);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.3);border-radius:8px;cursor:pointer;font:inherit;';
  helpClose.onclick = () => (help.style.display = 'none');
  helpCard.append(helpTitle, helpBody, helpClose);
  help.append(helpCard);
  help.addEventListener('click', (e) => {
    if (e.target === help) help.style.display = 'none';
  });
  document.body.appendChild(help);

  const helpToggle = document.createElement('button');
  helpToggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  helpToggle.onclick = () => (help.style.display = 'grid');
  ui.addTool(helpToggle);

  function relabelHelp(): void {
    helpTitle.textContent = t('helpTitle');
    helpBody.innerHTML = t('helpBody');
    helpToggle.textContent = '? ' + t('help');
  }
  relabelHelp();
  onLang(relabelHelp);

  // Show the explainer once on first visit.
  if (!localStorage.getItem('pelagia-seen-help')) {
    help.style.display = 'grid';
    localStorage.setItem('pelagia-seen-help', '1');
  }
  // --- Observatory: full-screen data view (world history + lineages + watch-list) ---
  const MAX_WORLD_SAMPLES = 240; // ring length for every time series
  const MAX_TRACKED_LINEAGES = 80;
  interface LinTrack {
    hue: number;
    samples: number[];
    lastTick: number;
    count: number;
    trend: number;
    descKey: string;
    fast: boolean;
    seek: number;
    forage: number;
    cruise: number;
    aggression: number;
    neurons: number;
  }
  const worldSeries: WorldSample[] = [];
  const lineageHist = new Map<number, LinTrack>();
  const watched = new Map<number, Watched>();
  let watchPending = false;

  const observatory = buildObservatory((id) => removeWatch(id));
  document.body.appendChild(observatory.panel);
  ui.addTool(observatory.toggle);
  const history = buildEvolutionHistory();
  document.body.appendChild(history.panel);
  ui.addTool(history.toggle);

  function addWatch(id: number, lineage: number): void {
    if (id < 0 || watched.has(id) || watched.size >= MAX_WATCH) return;
    watched.set(id, { id, lineage, hue: floatFromU32(pcgHash(lineage)), history: [] });
    pushObservatory();
  }
  function removeWatch(id: number): void {
    watched.delete(id);
    pushObservatory();
  }
  function buildObservatoryData(): ObservatoryData {
    const lineages: LineageHistory[] = [];
    for (const [lin, tr] of lineageHist) {
      lineages.push({
        lineage: lin,
        hue: tr.hue,
        count: tr.count,
        trend: tr.trend,
        descKey: tr.descKey,
        fast: tr.fast,
        seek: tr.seek,
        forage: tr.forage,
        cruise: tr.cruise,
        aggression: tr.aggression,
        neurons: tr.neurons,
        samples: tr.samples,
      });
    }
    return { world: worldSeries, lineages, watched: [...watched.values()] };
  }
  function pushObservatory(): void {
    const obsData = buildObservatoryData();
    observatory.update(obsData);
    history.update(obsData);
  }

  // Sample the tracked creatures through the inspect pass (one tiny dispatch each;
  // capped at MAX_WATCH). Records energy/speed/decision history per individual.
  async function sampleWatched(): Promise<void> {
    if (watchPending) return;
    const ids = [...watched.keys()];
    if (ids.length === 0) return;
    watchPending = true;
    try {
      for (let k = 0; k < ids.length; k++) {
        pu[22] = ids[k]!;
        device.queue.writeBuffer(paramsBuf, 0, pbuf);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(inspectPipeline);
        pass.setBindGroup(0, inspectBind);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(
          inspectBuf,
          0,
          watchReadback,
          k * INSPECT_FLOATS * 4,
          INSPECT_FLOATS * 4,
        );
        device.queue.submit([enc.finish()]);
      }
      setSelectedUniform(); // restore the live selection index for the brain view
      device.queue.writeBuffer(paramsBuf, 0, pbuf);
      await watchReadback.mapAsync(GPUMapMode.READ);
      const d = new Float32Array(watchReadback.getMappedRange().slice(0));
      watchReadback.unmap();
      ids.forEach((id, k) => {
        const w = watched.get(id);
        if (!w) return;
        const o = k * INSPECT_FLOATS;
        const sameLineage = Math.round(d[o + 26]!) === w.lineage;
        const sample: WatchSample = {
          tick: frame,
          energy: d[o + 24]!,
          speed: d[o + 23]!,
          turn: d[o + 18]!,
          thrust: (d[o + 19]! + 1) / 2,
          alive: d[o + 27]! >= 0.5 && sameLineage,
        };
        w.history.push(sample);
        if (w.history.length > MAX_WORLD_SAMPLES) w.history.shift();
      });
    } finally {
      watchPending = false;
    }
  }

  // Maintain per-clade population curves on ONE shared time axis (every tracked
  // clade gets exactly one sample per call, 0 when absent), so they can be stacked
  // into a Muller plot. Track newly-seen clades; bound the map (evict long-extinct).
  function updateLineageHistories(
    counts: Map<number, number>,
    poolMeta: Map<number, LineageTraits>,
  ): void {
    // 1. Start tracking newly-seen, sizeable/characterised clades.
    for (const [lin, count] of counts) {
      if (lineageHist.has(lin)) continue;
      const meta = poolMeta.get(lin);
      if (!meta && count < 2) continue;
      lineageHist.set(lin, {
        hue: floatFromU32(pcgHash(lin)),
        samples: [],
        lastTick: frame,
        count: 0,
        trend: 0,
        descKey: meta?.descKey ?? 'desc_erratic',
        fast: meta?.fast ?? false,
        seek: meta?.seek ?? 0,
        forage: meta?.forage ?? 0,
        cruise: meta?.cruise ?? 0,
        aggression: meta?.aggression ?? 0,
        neurons: meta?.neurons ?? 0,
      });
    }
    // 2. Append one aligned sample to every tracked clade + refresh its traits.
    for (const [lin, tr] of lineageHist) {
      const c = counts.get(lin) ?? 0;
      const meta = poolMeta.get(lin);
      if (meta) {
        tr.descKey = meta.descKey;
        tr.fast = meta.fast;
        tr.seek = meta.seek;
        tr.forage = meta.forage;
        tr.cruise = meta.cruise;
        tr.aggression = meta.aggression;
        tr.neurons = meta.neurons;
      }
      tr.trend = c - tr.count;
      tr.count = c;
      if (c > 0) tr.lastTick = frame;
      tr.samples.push(c);
      if (tr.samples.length > MAX_WORLD_SAMPLES) tr.samples.shift();
    }
    // 3. Bound the map (evict long-extinct, then smallest).
    if (lineageHist.size > MAX_TRACKED_LINEAGES) {
      const entries = [...lineageHist.entries()].sort((a, b) => {
        const da = a[1].count === 0 ? 0 : 1;
        const db = b[1].count === 0 ? 0 : 1;
        if (da !== db) return da - db; // dead clades first
        if (da === 0) return a[1].lastTick - b[1].lastTick; // oldest-dead first
        return a[1].count - b[1].count; // then smallest alive
      });
      for (const [lin] of entries) {
        if (lineageHist.size <= MAX_TRACKED_LINEAGES) break;
        lineageHist.delete(lin);
      }
    }
  }

  let analysisPending = false;
  let lastAnalysisT = 0;
  const prevCounts = new Map<number, number>();

  // Periodic world sample: reads bio + food back, drives the lineage panel AND the
  // observatory (world time series, lineage histories, tracked creatures).
  async function sampleWorld(): Promise<void> {
    if (analysisPending) return;
    analysisPending = true;
    try {
      const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
      const fRead = device.createBuffer({ size: f * 8, usage: GPUBufferUsage.MAP_READ | CD });
      let enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
      enc.copyBufferToBuffer(foodBuf, 0, fRead, 0, f * 8);
      device.queue.submit([enc.finish()]);
      await Promise.all([bRead.mapAsync(GPUMapMode.READ), fRead.mapAsync(GPUMapMode.READ)]);
      const bi = new Float32Array(bRead.getMappedRange().slice(0));
      const fo = new Float32Array(fRead.getMappedRange().slice(0));
      bRead.unmap();
      bRead.destroy();
      let foodAlive = 0;
      for (let j = 0; j < f; j++) if (fo[j * 2]! >= 0) foodAlive++;
      fRead.unmap();
      fRead.destroy();

      const counts = new Map<number, number>();
      const repSlot = new Map<number, number>();
      let aliveCount = 0;
      let sumE = 0;
      let minE = Infinity;
      let maxE = -Infinity;
      for (let i = 0; i < n; i++) {
        if (bi[i * 4 + 2]! < 0.5) continue;
        aliveCount++;
        const e = bi[i * 4]!;
        sumE += e;
        if (e < minE) minE = e;
        if (e > maxE) maxE = e;
        const lin = Math.round(bi[i * 4 + 3]!);
        counts.set(lin, (counts.get(lin) ?? 0) + 1);
        if (!repSlot.has(lin)) repSlot.set(lin, i);
      }
      const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
      const pool = ranked.slice(0, 16); // characterise a pool, then pick groups

      const rows: LineageRow[] = [];
      const strategy: Record<string, number> = {};
      const poolMeta = new Map<number, LineageTraits>();
      let meanSize = 1;
      if (pool.length > 0) {
        const gRead = device.createBuffer({
          size: pool.length * GENOME_SIZE * 4,
          usage: GPUBufferUsage.MAP_READ | CD,
        });
        enc = device.createCommandEncoder();
        pool.forEach(([lin], k) => {
          enc.copyBufferToBuffer(
            weightsBuf,
            repSlot.get(lin)! * GENOME_SIZE * 4,
            gRead,
            k * GENOME_SIZE * 4,
            GENOME_SIZE * 4,
          );
        });
        device.queue.submit([enc.finish()]);
        await gRead.mapAsync(GPUMapMode.READ);
        const gAll = new Float32Array(gRead.getMappedRange().slice(0));
        gRead.unmap();
        gRead.destroy();

        const traits = pool.map((_e, k) =>
          characterizeGenome(gAll.subarray(k * GENOME_SIZE, (k + 1) * GENOME_SIZE)),
        );
        const dominantCount = Math.min(5, pool.length);
        // Behavioural centroid of the dominant clades; the "distinct" section
        // surfaces the non-dominant survivors whose policy differs MOST from it
        // (real behavioural variety, not just the smallest clades — which, after
        // convergence, tend to be ordinary too).
        const axes = (tr: LineageTraits): number[] => [tr.seek, tr.forage, tr.cruise, tr.turnBias];
        const centroid = [0, 0, 0, 0];
        for (let k = 0; k < dominantCount; k++) {
          const a = axes(traits[k]!);
          for (let d = 0; d < 4; d++) centroid[d]! += a[d]! / dominantCount;
        }
        const dist = (a: number[]): number =>
          Math.hypot(
            a[0]! - centroid[0]!,
            a[1]! - centroid[1]!,
            a[2]! - centroid[2]!,
            a[3]! - centroid[3]!,
          );
        const distinctIdx: number[] = [];
        for (let k = dominantCount; k < pool.length; k++) distinctIdx.push(k);
        distinctIdx.sort((a, b) => dist(axes(traits[b]!)) - dist(axes(traits[a]!)));

        const pushRow = (k: number, group: 'dominant' | 'distinct'): void => {
          const [lin, count] = pool[k]!;
          const ch = traits[k]!;
          rows.push({
            lineage: lin,
            hue: floatFromU32(pcgHash(lin)),
            count,
            trend: count - (prevCounts.get(lin) ?? count),
            group,
            descKey: ch.descKey,
            fast: ch.fast,
            seek: ch.seek,
            forage: ch.forage,
            cruise: ch.cruise,
            aggression: ch.aggression,
            neurons: ch.neurons,
          });
        };
        for (let k = 0; k < dominantCount; k++) pushRow(k, 'dominant');
        for (const k of distinctIdx.slice(0, 3)) pushRow(k, 'distinct');

        prevCounts.clear();
        for (const [lin, count] of counts) prevCounts.set(lin, count);

        // Strategy mix snapshot + per-lineage trait metadata for the observatory,
        // plus a population-weighted mean body size for the narrative.
        let szSum = 0;
        let cSum = 0;
        pool.forEach(([lin, count], k) => {
          const tr = traits[k]!;
          strategy[tr.descKey] = (strategy[tr.descKey] ?? 0) + count;
          poolMeta.set(lin, tr);
          szSum += tr.size * count;
          cSum += count;
        });
        if (cSum > 0) meanSize = szSum / cSum;
      }
      lineagePanel.update(rows);

      // --- Observatory time series ---
      updateLineageHistories(counts, poolMeta);
      worldSeries.push({
        tick: frame,
        alive: aliveCount,
        foodAlive,
        lineages: counts.size,
        energyAvg: aliveCount ? sumE / aliveCount : 0,
        energyMin: aliveCount ? minE : 0,
        energyMax: aliveCount ? maxE : 0,
        predKills: lastPredKills,
        predGain: pf[24]!,
        meanSize,
        strategy,
      });
      if (worldSeries.length > MAX_WORLD_SAMPLES) worldSeries.shift();
      await sampleWatched();
      pushObservatory();
    } finally {
      analysisPending = false;
    }
  }

  // Occasional non-blocking readback of the alive count (= n - freeCount) and the
  // per-tick predation kill count (gridData[2*numCells .. +3] = free, budget, pred).
  let alive = n;
  let lastPredKills = 0;
  let countersPending = false;
  const countersReadback = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  function pollAlive(): void {
    if (countersPending) return;
    countersPending = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gridDataBuf, 2 * numCells * 4, countersReadback, 0, 12);
    device.queue.submit([enc.finish()]);
    void countersReadback.mapAsync(GPUMapMode.READ).then(() => {
      const c = new Uint32Array(countersReadback.getMappedRange());
      alive = n - c[0]!;
      lastPredKills = c[2]!;
      countersReadback.unmap();
      countersPending = false;
    });
  }

  // --- Chemotaxis validation hook (read GPU state back, compute on CPU) ---
  (globalThis as unknown as { __pelagia: unknown }).__pelagia = {
    n,
    f,
    seed,
    get tick() {
      return frame;
    },
    readChemotaxis,
  };
  async function readChemotaxis(): Promise<unknown> {
    const sRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const fRead = device.createBuffer({ size: f * 8, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(stateBuf, 0, sRead, 0, n * 16);
    enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
    enc.copyBufferToBuffer(foodBuf, 0, fRead, 0, f * 8);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      sRead.mapAsync(GPUMapMode.READ),
      bRead.mapAsync(GPUMapMode.READ),
      fRead.mapAsync(GPUMapMode.READ),
    ]);
    const st = new Float32Array(sRead.getMappedRange().slice(0));
    const bi = new Float32Array(bRead.getMappedRange().slice(0));
    const fo = new Float32Array(fRead.getMappedRange().slice(0));
    sRead.unmap();
    bRead.unmap();
    fRead.unmap();

    const fx = new Float32Array(f);
    const fy = new Float32Array(f);
    for (let j = 0; j < f; j++) {
      fx[j] = fo[j * 2]!;
      fy[j] = fo[j * 2 + 1]!;
    }
    const grid = new SpatialGrid(world, world, cellSize);
    grid.build(fx, fy, f);

    let count = 0;
    let sumCos = 0;
    let seeking = 0;
    let aliveCount = 0;
    let sumE = 0;
    let minE = Infinity;
    let maxE = -Infinity;
    for (let i = 0; i < n; i++) {
      if (bi[i * 4 + 2]! < 0.5) continue;
      aliveCount++;
      const e = bi[i * 4]!;
      sumE += e;
      if (e < minE) minE = e;
      if (e > maxE) maxE = e;
      const px = st[i * 4]!;
      const py = st[i * 4 + 1]!;
      const heading = st[i * 4 + 2]!;
      const fj = grid.findNearest(px, py, cellSize);
      if (fj < 0) continue;
      const dx = wrapDelta(fx[fj]! - px, world);
      const dy = wrapDelta(fy[fj]! - py, world);
      const c = Math.cos(Math.atan2(dy, dx) - heading);
      sumCos += c;
      if (c > 0.5) seeking++;
      count++;
    }
    let aliveFood = 0;
    for (let j = 0; j < f; j++) if (fx[j]! >= 0) aliveFood++;
    return {
      tick: frame,
      c0: {
        x: +st[0]!.toFixed(1),
        y: +st[1]!.toFixed(1),
        heading: +st[2]!.toFixed(2),
        energy: bi[0]!,
        lineage: bi[3]!,
      },
      alive: aliveCount,
      aliveFood,
      withFood: count,
      energy: {
        min: +minE.toFixed(1),
        avg: +(sumE / Math.max(1, aliveCount)).toFixed(1),
        max: +maxE.toFixed(1),
      },
      chemotaxis: +(sumCos / Math.max(1, count)).toFixed(3),
      fracSeeking: +(seeking / Math.max(1, count)).toFixed(3),
    };
  }

  let frame = 0;
  let frames = 0;
  let stepAcc = 0;
  let lastFpsT = performance.now();

  // Restart the ocean in place: reseed the buffers from the current seed, empty
  // the free-slot stack, rewind the frame counter (so the shader RNG replays),
  // wipe trails and drop any selection. Same seed + current god params = a
  // reproducible re-run; revives everyone after an extinction without a reload.
  function resetOcean(): void {
    fillInitialData();
    device.queue.writeBuffer(stateBuf, 0, stateData);
    device.queue.writeBuffer(bioBuf, 0, bioData);
    device.queue.writeBuffer(weightsBuf, 0, weightsData);
    device.queue.writeBuffer(foodBuf, 0, foodData);
    device.queue.writeBuffer(gridDataBuf, 2 * numCells * 4, new Uint32Array([0])); // freeCount = 0
    frame = 0;
    stepAcc = 0;
    alive = n;
    prevCounts.clear();
    lineagePanel.update([]);
    deselect();
    const enc = device.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: accumView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      .end();
    device.queue.submit([enc.finish()]);
  }

  // Headless warm-up: advance the simulation before the first rendered frame so
  // the ocean lands in a chosen state (past the initial die-off, mid-evolution).
  const warmup = opts.warmup ?? 0;
  for (let w = 0; w < warmup; w++) {
    writeParams(frame);
    const enc = device.createCommandEncoder();
    recordSim(enc);
    device.queue.submit([enc.finish()]);
    frame++;
  }
  opts.onReady?.();

  function tick(): void {
    // Keep the drawing buffer in sync with the viewport (handles a late initial
    // layout where innerWidth was 0, and any window resize).
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const wantW = Math.max(1, Math.floor(window.innerWidth * dpr));
    const wantH = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (canvas.width !== wantW || canvas.height !== wantH) resize();

    // Simulation steps. Speed may be < 1 (slow motion): accumulate fractional
    // steps so e.g. 0.5x runs one tick every other frame. Each step is its own
    // submit so the frame counter (used for RNG keying) varies per tick.
    let steps = 0;
    if (!ui.paused) {
      stepAcc += ui.speed;
      steps = Math.floor(stepAcc);
      stepAcc -= steps;
    }
    for (let k = 0; k < steps; k++) {
      writeParams(frame);
      const enc = device.createCommandEncoder();
      recordSim(enc);
      device.queue.submit([enc.finish()]);
      frame++;
    }

    // Render every frame (so trails keep fading even while paused).
    const encoder = device.createCommandEncoder();
    const accumPass = encoder.beginRenderPass({
      colorAttachments: [{ view: accumView, loadOp: 'load', storeOp: 'store' }],
    });
    accumPass.setPipeline(fadePipeline);
    accumPass.draw(3);
    accumPass.setPipeline(foodPipeline);
    accumPass.setBindGroup(0, foodBind);
    accumPass.draw(6, f);
    accumPass.setPipeline(creaturePipeline);
    accumPass.setBindGroup(0, creatureBind);
    accumPass.draw(6, n);
    accumPass.end();

    const view = context!.getCurrentTexture().createView();
    const present = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    present.setPipeline(presentPipeline);
    present.setBindGroup(0, presentBind);
    present.draw(3);
    present.end();
    device.queue.submit([encoder.finish()]);

    // Brain inspector: recompute the selected creature's brain and read it back.
    if (selectedIndex >= 0 && !inspectPending) {
      setSelectedUniform();
      writeParams(frame); // includes pu[22] = selectedIndex
      const inspectFrame = frame; // tag the sample so the decision tape advances per tick
      const ienc = device.createCommandEncoder();
      const ipass = ienc.beginComputePass();
      ipass.setPipeline(inspectPipeline);
      ipass.setBindGroup(0, inspectBind);
      ipass.dispatchWorkgroups(1);
      ipass.end();
      ienc.copyBufferToBuffer(inspectBuf, 0, inspectReadback, 0, INSPECT_FLOATS * 4);
      device.queue.submit([ienc.finish()]);
      inspectPending = true;
      void inspectReadback.mapAsync(GPUMapMode.READ).then(() => {
        const d = new Float32Array(inspectReadback.getMappedRange().slice(0));
        inspectReadback.unmap();
        inspectPending = false;
        if (selectedIndex < 0) return;
        const alive = d[27]! >= 0.5;
        const sameLineage = Math.round(d[26]!) === selectedLineage;
        if (alive && sameLineage) {
          brainView.update(d, inspectFrame);
          positionRing(d);
        } else if (!alive && sameLineage) {
          // Our creature died: show it deceased once, then stop following.
          brainView.update(d, inspectFrame);
          ring.style.display = 'none';
          selectedIndex = -1;
          setSelectedUniform();
        } else {
          // The slot was reused by a different creature — drop the selection
          // rather than silently follow the impostor.
          deselect();
        }
      });
    }

    frames++;
    const now = performance.now();
    if (now - lastFpsT >= 500) {
      const fps = (frames * 1000) / (now - lastFpsT);
      frames = 0;
      lastFpsT = now;
      pollAlive();
      ui.update(alive, fps, frame);
      if (now - lastAnalysisT > 1500) {
        lastAnalysisT = now;
        void sampleWorld();
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
