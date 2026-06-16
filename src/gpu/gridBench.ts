import { initGpu } from './device.js';
import gridCommon from './shaders/grid_common.wgsl?raw';
import gridPasses from './shaders/grid_passes.wgsl?raw';
import gridSim from './shaders/grid_sim.wgsl?raw';
import renderShader from './shaders/render.wgsl?raw';
import { DEFAULT_CONFIG } from '../core/config.js';
import { GENOME_SIZE } from '../sim/brain.js';
import { SpatialGrid } from '../sim/grid.js';
import { toroidalDistSq } from '../core/space.js';

/**
 * Phase 1.2: the realistic per-tick workload on the GPU — spatial grid build
 * (counting sort) + nearest-food sensing + brain + move. World size scales with
 * N to hold density roughly constant, so this measures "N creatures at a
 * realistic density at 60fps", which is the true performance ceiling.
 *
 * On startup it validates the grid+sensing against the CPU SpatialGrid oracle.
 */
export async function runGpuGrid(canvas: HTMLCanvasElement, requestedN: number): Promise<void> {
  const { device, format } = await initGpu();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('webgpu canvas context unavailable');
  context.configure({ device, format, alphaMode: 'opaque' });

  const cellSize = DEFAULT_CONFIG.perceptionRadius;
  const maxN = Math.floor(device.limits.maxStorageBufferBindingSize / (GENOME_SIZE * 4));
  const n = Math.min(requestedN, maxN);
  const f = n; // one food pellet per creature
  const spacing = 40; // world units per creature -> density
  const world = Math.max(cellSize * 3, spacing * Math.sqrt(n));
  const cols = Math.ceil(world / cellSize);
  const rows = Math.ceil(world / cellSize);
  const numCells = cols * rows;

  // --- Initial data (kept on CPU for oracle validation) ---
  const stateData = new Float32Array(n * 4);
  const hueData = new Float32Array(n);
  const foodData = new Float32Array(f * 2);
  for (let i = 0; i < n; i++) {
    stateData[i * 4 + 0] = Math.random() * world;
    stateData[i * 4 + 1] = Math.random() * world;
    stateData[i * 4 + 2] = Math.random() * Math.PI * 2;
    hueData[i] = Math.random();
  }
  for (let j = 0; j < f; j++) {
    foodData[j * 2 + 0] = Math.random() * world;
    foodData[j * 2 + 1] = Math.random() * world;
  }
  const weightsData = new Float32Array(n * GENOME_SIZE);
  for (let i = 0; i < weightsData.length; i++) weightsData[i] = Math.random() * 2 - 1;

  // --- Buffers ---
  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const stateBuf = device.createBuffer({ size: stateData.byteLength, usage: S | CD });
  const hueBuf = device.createBuffer({ size: hueData.byteLength, usage: S | CD });
  const foodBuf = device.createBuffer({ size: foodData.byteLength, usage: S | CD });
  const weightsBuf = device.createBuffer({ size: weightsData.byteLength, usage: S | CD });
  const cellCountBuf = device.createBuffer({ size: numCells * 4, usage: S | CD });
  const cellStartBuf = device.createBuffer({ size: (numCells + 1) * 4, usage: S | CD });
  const cursorBuf = device.createBuffer({ size: numCells * 4, usage: S });
  const sortedBuf = device.createBuffer({ size: f * 4, usage: S });
  const nearestBuf = device.createBuffer({ size: n * 4, usage: S | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(stateBuf, 0, stateData);
  device.queue.writeBuffer(hueBuf, 0, hueData);
  device.queue.writeBuffer(foodBuf, 0, foodData);
  device.queue.writeBuffer(weightsBuf, 0, weightsData);

  const gridUbo = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | CD });
  const gp = new ArrayBuffer(64);
  const gpf = new Float32Array(gp);
  const gpu32 = new Uint32Array(gp);
  gpf[0] = world;
  gpf[1] = world;
  gpf[2] = cellSize;
  gpf[3] = DEFAULT_CONFIG.maxSpeed;
  gpf[4] = DEFAULT_CONFIG.maxTurnRate;
  gpf[5] = 1; // dt
  gpu32[8] = cols;
  gpu32[9] = rows;
  gpu32[10] = numCells;
  gpu32[11] = n;
  gpu32[12] = f;
  device.queue.writeBuffer(gridUbo, 0, gp);

  // --- Shared bind group layout for all grid passes ---
  const storage: GPUBufferBindingLayout = { type: 'storage' };
  const readonly: GPUBufferBindingLayout = { type: 'read-only-storage' };
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: storage },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: storage },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: storage },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: storage },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: storage },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: storage },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const bind = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: gridUbo } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: weightsBuf } },
      { binding: 3, resource: { buffer: foodBuf } },
      { binding: 4, resource: { buffer: cellCountBuf } },
      { binding: 5, resource: { buffer: cellStartBuf } },
      { binding: 6, resource: { buffer: cursorBuf } },
      { binding: 7, resource: { buffer: sortedBuf } },
      { binding: 8, resource: { buffer: nearestBuf } },
    ],
  });

  const passesModule = device.createShaderModule({ code: gridCommon + '\n' + gridPasses });
  const simModule = device.createShaderModule({ code: gridCommon + '\n' + gridSim });
  const mk = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
    device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
  const pClear = mk(passesModule, 'clear');
  const pCount = mk(passesModule, 'count');
  const pScan = mk(passesModule, 'scan');
  const pScatter = mk(passesModule, 'scatter');
  const pSim = mk(simModule, 'main');

  function recordSim(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bind);
    pass.setPipeline(pClear);
    pass.dispatchWorkgroups(Math.ceil(numCells / 256));
    pass.setPipeline(pCount);
    pass.dispatchWorkgroups(Math.ceil(f / 256));
    pass.setPipeline(pScan);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pScatter);
    pass.dispatchWorkgroups(Math.ceil(f / 256));
    pass.setPipeline(pSim);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();
  }

  // --- Validate the grid+sensing against the CPU oracle ---
  await validate();
  async function validate(): Promise<void> {
    const readback = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    recordSim(enc);
    enc.copyBufferToBuffer(nearestBuf, 0, readback, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const gpuNearest = new Int32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const fx = new Float32Array(f);
    const fy = new Float32Array(f);
    for (let j = 0; j < f; j++) {
      fx[j] = foodData[j * 2]!;
      fy[j] = foodData[j * 2 + 1]!;
    }
    const cpuGrid = new SpatialGrid(world, world, cellSize);
    cpuGrid.build(fx, fy, f);

    let mismatch = 0;
    const sample = Math.min(n, 5000); // sample for speed at large N
    for (let i = 0; i < sample; i++) {
      const px = stateData[i * 4]!;
      const py = stateData[i * 4 + 1]!;
      const cpuIdx = cpuGrid.findNearest(px, py, cellSize);
      const gpuIdx = gpuNearest[i]!;
      const cpuD =
        cpuIdx >= 0
          ? Math.sqrt(toroidalDistSq(px, py, fx[cpuIdx]!, fy[cpuIdx]!, world, world))
          : -1;
      const gpuD =
        gpuIdx >= 0
          ? Math.sqrt(toroidalDistSq(px, py, fx[gpuIdx]!, fy[gpuIdx]!, world, world))
          : -1;
      // Match if both empty, or both found a food at the same nearest distance
      // (index may differ on exact ties; compare distance to tolerate f32 vs f64).
      const ok =
        (cpuIdx < 0 && gpuIdx < 0) || (cpuIdx >= 0 && gpuIdx >= 0 && Math.abs(cpuD - gpuD) < 0.5);
      if (!ok) mismatch++;
    }
    const pct = (((sample - mismatch) / sample) * 100).toFixed(2);
    console.log(
      `[grid validation] N=${n} F=${f} numCells=${numCells} sample=${sample} match=${pct}% mismatches=${mismatch}`,
    );
  }

  // --- Render pipeline (creatures, instanced; zero-copy from state) ---
  const renderUbo = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | CD });
  const renderData = new Float32Array(8);
  const renderModule = device.createShaderModule({ code: renderShader });
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: {
      module: renderModule,
      entryPoint: 'fs',
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
  });
  const renderBind = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUbo } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: hueBuf } },
    ],
  });

  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.floor(window.innerWidth * dpr);
    const ch = Math.floor(window.innerHeight * dpr);
    canvas.width = cw;
    canvas.height = ch;
    const s = Math.min(cw / world, ch / world);
    renderData[0] = (s / cw) * 2;
    renderData[1] = -(s / ch) * 2;
    renderData[2] = ((cw - world * s) / 2 / cw) * 2 - 1;
    renderData[3] = 1 - ((ch - world * s) / 2 / ch) * 2;
    renderData[4] = Math.max(5, world / 300); // pointSize scales with world
    renderData[5] = 0.9;
    device.queue.writeBuffer(renderUbo, 0, renderData);
  }
  resize();
  window.addEventListener('resize', resize);

  const hud = document.createElement('div');
  hud.style.cssText =
    'position:fixed;top:8px;left:8px;padding:8px 12px;font:13px ui-monospace,monospace;' +
    'color:#cfe8ff;background:rgba(2,4,10,0.55);border-radius:6px;white-space:pre;pointer-events:none';
  document.body.appendChild(hud);

  let frame = 0;
  let frames = 0;
  let lastFpsT = performance.now();

  function tick(): void {
    gpu32[13] = frame;
    device.queue.writeBuffer(gridUbo, 0, gp);

    const encoder = device.createCommandEncoder();
    recordSim(encoder);
    const view = context!.getCurrentTexture().createView();
    const rpass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.008, g: 0.016, b: 0.04, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    rpass.setPipeline(renderPipeline);
    rpass.setBindGroup(0, renderBind);
    rpass.draw(3, n);
    rpass.end();
    device.queue.submit([encoder.finish()]);

    frame++;
    frames++;
    const now = performance.now();
    if (now - lastFpsT >= 500) {
      const fps = (frames * 1000) / (now - lastFpsT);
      frames = 0;
      lastFpsT = now;
      hud.textContent = `PELAGIA · GPU spike (grid)\ncreatures ${n.toLocaleString()}\nfood ${f.toLocaleString()} · cells ${numCells.toLocaleString()}\nFPS ${fps.toFixed(0)}`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
