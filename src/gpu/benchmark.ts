import { initGpu } from './device.js';
import moveShader from './shaders/move.wgsl?raw';
import brainShader from './shaders/brain_move.wgsl?raw';
import renderShader from './shaders/render.wgsl?raw';
import { DEFAULT_CONFIG } from '../core/config.js';
import { GENOME_SIZE } from '../sim/brain.js';

export type BenchMode = 'move' | 'brain';

/**
 * Phase 1.0/1.1 performance spike: N creatures on the GPU, drawn with one
 * instanced triangle each (state read straight from the GPU buffer — no
 * readback). `mode` selects the per-tick compute:
 *   - 'move':  a cheap wander (render + buffer ceiling).
 *   - 'brain': a full MLP forward pass per creature (the brain-eval cost).
 * Reports FPS so we can find the ceiling before adding the spatial grid (1.2).
 */
export async function runGpuBenchmark(
  canvas: HTMLCanvasElement,
  requestedN: number,
  mode: BenchMode,
): Promise<void> {
  const { device, format } = await initGpu();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('webgpu canvas context unavailable');
  context.configure({ device, format, alphaMode: 'opaque' });

  const width = DEFAULT_CONFIG.width;
  const height = DEFAULT_CONFIG.height;

  // In brain mode the genome buffer dominates memory; cap N to the binding limit.
  let n = requestedN;
  if (mode === 'brain') {
    const maxN = Math.floor(device.limits.maxStorageBufferBindingSize / (GENOME_SIZE * 4));
    n = Math.min(n, maxN);
  }

  // --- Buffers ---
  const stateData = new Float32Array(n * 4);
  const hueData = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    stateData[i * 4 + 0] = Math.random() * width;
    stateData[i * 4 + 1] = Math.random() * height;
    stateData[i * 4 + 2] = Math.random() * Math.PI * 2;
    hueData[i] = Math.random();
  }
  const stateBuf = device.createBuffer({
    size: stateData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(stateBuf, 0, stateData);
  const hueBuf = device.createBuffer({
    size: hueData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(hueBuf, 0, hueData);

  let weightsBuf: GPUBuffer | undefined;
  if (mode === 'brain') {
    const weightsData = new Float32Array(n * GENOME_SIZE);
    for (let i = 0; i < weightsData.length; i++) weightsData[i] = (Math.random() * 2 - 1) * 1.0;
    weightsBuf = device.createBuffer({
      size: weightsData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(weightsBuf, 0, weightsData);
  }

  const params = new ArrayBuffer(32);
  const pf = new Float32Array(params);
  const pu = new Uint32Array(params);
  pf[0] = width;
  pf[1] = height;
  pf[2] = 1; // dt
  pf[3] = DEFAULT_CONFIG.maxSpeed;
  pf[4] = DEFAULT_CONFIG.maxTurnRate;
  pu[6] = n;
  const paramsUbo = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const renderUbo = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const renderData = new Float32Array(8);

  // --- Compute pipeline (mode-dependent) ---
  const computeModule = device.createShaderModule({
    code: mode === 'brain' ? brainShader : moveShader,
  });
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'main' },
  });
  const computeEntries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: paramsUbo } },
    { binding: 1, resource: { buffer: stateBuf } },
  ];
  if (weightsBuf) computeEntries.push({ binding: 2, resource: { buffer: weightsBuf } });
  const computeBind = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: computeEntries,
  });

  // --- Render pipeline ---
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

  // --- View transform (world -> NDC, aspect-preserving, centred) ---
  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.floor(window.innerWidth * dpr);
    const ch = Math.floor(window.innerHeight * dpr);
    canvas.width = cw;
    canvas.height = ch;
    const s = Math.min(cw / width, ch / height);
    renderData[0] = (s / cw) * 2;
    renderData[1] = -(s / ch) * 2;
    renderData[2] = ((cw - width * s) / 2 / cw) * 2 - 1;
    renderData[3] = 1 - ((ch - height * s) / 2 / ch) * 2;
    renderData[4] = 5; // pointSize
    renderData[5] = 0.9; // brightness
    device.queue.writeBuffer(renderUbo, 0, renderData);
  }
  resize();
  window.addEventListener('resize', resize);

  // --- FPS overlay ---
  const hud = document.createElement('div');
  hud.style.cssText =
    'position:fixed;top:8px;left:8px;padding:8px 12px;font:13px ui-monospace,monospace;' +
    'color:#cfe8ff;background:rgba(2,4,10,0.55);border-radius:6px;white-space:pre;pointer-events:none';
  document.body.appendChild(hud);

  let frame = 0;
  let frames = 0;
  let lastFpsT = performance.now();
  const workgroups = Math.ceil(n / 256);

  function tick(): void {
    pu[5] = frame;
    device.queue.writeBuffer(paramsUbo, 0, params);

    const encoder = device.createCommandEncoder();
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(computePipeline);
    cpass.setBindGroup(0, computeBind);
    cpass.dispatchWorkgroups(workgroups);
    cpass.end();

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
      hud.textContent = `PELAGIA · GPU spike (${mode})\ncreatures ${n.toLocaleString()}\nFPS ${fps.toFixed(0)}`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
