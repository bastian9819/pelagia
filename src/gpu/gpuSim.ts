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
import { GENOME_SIZE } from '../sim/brain.js';
import { pcgHash, floatFromU32 } from '../core/rng.js';
import { SpatialGrid } from '../sim/grid.js';
import { wrapDelta } from '../core/space.js';
import { buildUi } from './ui.js';
import { buildBrainView } from './brainView.js';
import { buildLineagePanel, characterizeGenome, type LineageRow } from './lineages.js';
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
  const requestedN = opts.n;
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

  // --- Initial data ---
  const stateData = new Float32Array(n * 4);
  const bioData = new Float32Array(n * 4);
  const weightsData = new Float32Array(n * GENOME_SIZE);
  const foodData = new Float32Array(f * 2);
  for (let i = 0; i < n; i++) {
    stateData[i * 4 + 0] = Math.random() * world;
    stateData[i * 4 + 1] = Math.random() * world;
    stateData[i * 4 + 2] = Math.random() * Math.PI * 2;
    bioData[i * 4 + 0] = cfg.initialEnergy;
    bioData[i * 4 + 1] = floatFromU32(pcgHash(i)); // hue derived from lineage id
    bioData[i * 4 + 2] = 1; // alive
    bioData[i * 4 + 3] = i; // lineage id (= founder slot; inherited unchanged)
  }
  for (let i = 0; i < weightsData.length; i++) {
    weightsData[i] = (Math.random() * 2 - 1) * cfg.weightInitStd;
  }
  for (let j = 0; j < f; j++) {
    if (j < foodInitAlive) {
      foodData[j * 2 + 0] = Math.random() * world;
      foodData[j * 2 + 1] = Math.random() * world;
    } else {
      foodData[j * 2 + 0] = -1; // dead slot (sentinel)
      foodData[j * 2 + 1] = -1;
    }
  }

  // --- Buffers ---
  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const CS = GPUBufferUsage.COPY_SRC;
  const stateBuf = device.createBuffer({ size: stateData.byteLength, usage: S | CD | CS });
  const bioBuf = device.createBuffer({ size: bioData.byteLength, usage: S | CD | CS });
  const weightsBuf = device.createBuffer({ size: weightsData.byteLength, usage: S | CD | CS });
  const foodBuf = device.createBuffer({ size: foodData.byteLength, usage: S | CD | CS });
  // Packed atomic buffer: [cell counts/cursor | freeCount | budget | claims].
  const gridDataBuf = device.createBuffer({ size: (numCells + 2 + f) * 4, usage: S | CS });
  const cellStartBuf = device.createBuffer({ size: (numCells + 1) * 4, usage: S });
  const sortedBuf = device.createBuffer({ size: f * 4, usage: S });
  const freeListBuf = device.createBuffer({ size: n * 4, usage: S });
  // Brain-inspector output (one selected creature): inputs|hidden|outputs|state.
  const INSPECT_FLOATS = 32;
  const inspectBuf = device.createBuffer({ size: INSPECT_FLOATS * 4, usage: S | CS });
  const inspectReadback = device.createBuffer({
    size: INSPECT_FLOATS * 4,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  device.queue.writeBuffer(stateBuf, 0, stateData);
  device.queue.writeBuffer(bioBuf, 0, bioData);
  device.queue.writeBuffer(weightsBuf, 0, weightsData);
  device.queue.writeBuffer(foodBuf, 0, foodData);

  // --- Params uniform (96 bytes: 4 vec4<f32> + 2 vec4<u32>) ---
  const paramsBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | CD });
  const pbuf = new ArrayBuffer(96);
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
  const pScan = mk(gridModule, 'scan');
  const pScatter = mk(gridModule, 'scatter');
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
  const wgC = Math.ceil(numCells / 256);
  function recordSim(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, group0);
    pass.setBindGroup(1, group1);
    pass.setPipeline(pGridClear);
    pass.dispatchWorkgroups(wgC);
    pass.setPipeline(pClaimClear);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pCount);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pScan);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pScatter);
    pass.dispatchWorkgroups(wgF);
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

  // --- Stats panel + controls ---
  const ui = buildUi(() => {
    cam.zoom = 1;
    cam.cx = world / 2;
    cam.cy = world / 2;
    updateView();
  });
  document.body.appendChild(ui.panel);
  document.body.appendChild(ui.controls);

  // --- Selection + brain inspector ---
  let selectedIndex = -1;
  let selecting = false;
  let inspectPending = false;
  const brainView = buildBrainView(() => deselect());
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
    setSelectedUniform();
    brainView.hide();
    ring.style.display = 'none';
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
      setSelectedUniform();
      brainView.show();
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
  ui.controls.appendChild(lineagePanel.toggle);
  let analysisPending = false;
  let lastAnalysisT = 0;
  const prevCounts = new Map<number, number>();

  async function analyzeLineages(): Promise<void> {
    if (analysisPending) return;
    analysisPending = true;
    try {
      const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
      let enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
      device.queue.submit([enc.finish()]);
      await bRead.mapAsync(GPUMapMode.READ);
      const bi = new Float32Array(bRead.getMappedRange().slice(0));
      bRead.unmap();
      bRead.destroy();

      const counts = new Map<number, number>();
      const repSlot = new Map<number, number>();
      for (let i = 0; i < n; i++) {
        if (bi[i * 4 + 2]! < 0.5) continue;
        const lin = Math.round(bi[i * 4 + 3]!);
        counts.set(lin, (counts.get(lin) ?? 0) + 1);
        if (!repSlot.has(lin)) repSlot.set(lin, i);
      }
      const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);

      const rows: LineageRow[] = [];
      if (top.length > 0) {
        const gRead = device.createBuffer({
          size: top.length * GENOME_SIZE * 4,
          usage: GPUBufferUsage.MAP_READ | CD,
        });
        enc = device.createCommandEncoder();
        top.forEach(([lin], k) => {
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

        top.forEach(([lin, count], k) => {
          const genome = gAll.subarray(k * GENOME_SIZE, (k + 1) * GENOME_SIZE);
          rows.push({
            lineage: lin,
            hue: floatFromU32(pcgHash(lin)),
            count,
            trend: count - (prevCounts.get(lin) ?? count),
            desc: characterizeGenome(genome),
          });
        });
        prevCounts.clear();
        for (const [lin, count] of counts) prevCounts.set(lin, count);
      }
      lineagePanel.update(rows);
    } finally {
      analysisPending = false;
    }
  }

  // Occasional non-blocking readback of the alive count (= n - freeCount).
  let alive = n;
  let countersPending = false;
  const countersReadback = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  function pollAlive(): void {
    if (countersPending) return;
    countersPending = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gridDataBuf, numCells * 4, countersReadback, 0, 4);
    device.queue.submit([enc.finish()]);
    void countersReadback.mapAsync(GPUMapMode.READ).then(() => {
      const free = new Uint32Array(countersReadback.getMappedRange())[0]!;
      alive = n - free;
      countersReadback.unmap();
      countersPending = false;
    });
  }

  // --- Chemotaxis validation hook (read GPU state back, compute on CPU) ---
  (globalThis as unknown as { __pelagia: unknown }).__pelagia = {
    n,
    f,
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
        age: bi[3]!,
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
  let lastFpsT = performance.now();

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

    // Simulation steps (paused -> 0). Each step is its own submit so the frame
    // counter (used for RNG keying) varies per tick.
    const steps = ui.paused ? 0 : ui.speed;
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
        brainView.update(d);
        if (d[27]! < 0.5) {
          // The creature died: stop tracking but leave the panel on its last state.
          ring.style.display = 'none';
          selectedIndex = -1;
          setSelectedUniform();
        } else {
          positionRing(d);
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
        void analyzeLineages();
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
