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
import { SpatialGrid } from '../sim/grid.js';
import { wrapDelta } from '../core/space.js';

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
    bioData[i * 4 + 1] = Math.random(); // hue
    bioData[i * 4 + 2] = 1; // alive
    bioData[i * 4 + 3] = 0; // age
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
  const weightsBuf = device.createBuffer({ size: weightsData.byteLength, usage: S | CD });
  const foodBuf = device.createBuffer({ size: foodData.byteLength, usage: S | CD | CS });
  // Packed atomic buffer: [cell counts/cursor | freeCount | budget | claims].
  const gridDataBuf = device.createBuffer({ size: (numCells + 2 + f) * 4, usage: S | CS });
  const cellStartBuf = device.createBuffer({ size: (numCells + 1) * 4, usage: S });
  const sortedBuf = device.createBuffer({ size: f * 4, usage: S });
  const freeListBuf = device.createBuffer({ size: n * 4, usage: S });
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
    const cw = Math.floor(window.innerWidth * dpr);
    const ch = Math.floor(window.innerHeight * dpr);
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

    const s = Math.min(cw / world, ch / world);
    renderData[0] = (s / cw) * 2;
    renderData[1] = -(s / ch) * 2;
    renderData[2] = ((cw - world * s) / 2 / cw) * 2 - 1;
    renderData[3] = 1 - ((ch - world * s) / 2 / ch) * 2;
    renderData[4] = Math.max(cellSize * 0.22, world / 220); // glow radius (world units)
    renderData[5] = 1.4; // brightness (HDR; tonemapped on present)
    device.queue.writeBuffer(renderUbo, 0, renderData);
  }
  resize();
  window.addEventListener('resize', resize);

  const hud = document.createElement('div');
  hud.style.cssText =
    'position:fixed;top:8px;left:8px;padding:8px 12px;font:13px ui-monospace,monospace;' +
    'color:#cfe8ff;background:rgba(2,4,10,0.55);border-radius:6px;white-space:pre;pointer-events:none';
  document.body.appendChild(hud);

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
    writeParams(frame);
    const encoder = device.createCommandEncoder();
    recordSim(encoder);

    // Accumulation pass: fade previous frame (trails) -> food -> creatures.
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

    // Present pass: tonemap the accumulation texture to the swapchain.
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

    frame++;
    frames++;
    const now = performance.now();
    if (now - lastFpsT >= 500) {
      const fps = (frames * 1000) / (now - lastFpsT);
      frames = 0;
      lastFpsT = now;
      pollAlive();
      hud.textContent = `PELAGIA\n${alive.toLocaleString()} creatures alive\ntick ${frame.toLocaleString()} · ${fps.toFixed(0)} fps`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
