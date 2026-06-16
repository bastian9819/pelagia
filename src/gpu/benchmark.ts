import { initGpu } from './device.js';
import moveShader from './shaders/move.wgsl?raw';
import renderShader from './shaders/render.wgsl?raw';
import { DEFAULT_CONFIG } from '../core/config.js';

/**
 * Phase 1.0 performance spike: N creatures wandering on the GPU, updated by a
 * compute shader and drawn with one instanced triangle each (state read
 * straight from the GPU buffer — no readback). Reports FPS so we can find the
 * render+compute ceiling before adding brains and the spatial grid.
 */
export async function runGpuBenchmark(canvas: HTMLCanvasElement, n: number): Promise<void> {
  const { device, format } = await initGpu();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('webgpu canvas context unavailable');
  context.configure({ device, format, alphaMode: 'opaque' });

  const width = DEFAULT_CONFIG.width;
  const height = DEFAULT_CONFIG.height;

  // --- Buffers ---
  const stateData = new Float32Array(n * 4);
  const hueData = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    stateData[i * 4 + 0] = Math.random() * width;
    stateData[i * 4 + 1] = Math.random() * height;
    stateData[i * 4 + 2] = Math.random() * Math.PI * 2;
    stateData[i * 4 + 3] = 0;
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

  const moveParams = new ArrayBuffer(32);
  const moveF32 = new Float32Array(moveParams);
  const moveU32 = new Uint32Array(moveParams);
  moveF32[0] = width;
  moveF32[1] = height;
  moveF32[2] = 1; // dt
  moveF32[3] = DEFAULT_CONFIG.maxSpeed;
  moveF32[4] = DEFAULT_CONFIG.maxTurnRate;
  moveU32[6] = n;
  const moveUbo = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const renderUbo = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const renderData = new Float32Array(8);

  // --- Pipelines ---
  const moveModule = device.createShaderModule({ code: moveShader });
  const movePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: moveModule, entryPoint: 'main' },
  });
  const moveBind = device.createBindGroup({
    layout: movePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: moveUbo } },
      { binding: 1, resource: { buffer: stateBuf } },
    ],
  });

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
    const oxPx = (cw - width * s) / 2;
    const oyPx = (ch - height * s) / 2;
    renderData[0] = (s / cw) * 2; // sx
    renderData[1] = -(s / ch) * 2; // sy (flip)
    renderData[2] = (oxPx / cw) * 2 - 1; // ox
    renderData[3] = 1 - (oyPx / ch) * 2; // oy
    renderData[4] = 5; // pointSize (world units)
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
  let fps = 0;
  const workgroups = Math.ceil(n / 256);

  function tick(): void {
    moveU32[5] = frame;
    device.queue.writeBuffer(moveUbo, 0, moveParams);

    const encoder = device.createCommandEncoder();
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(movePipeline);
    cpass.setBindGroup(0, moveBind);
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
      fps = (frames * 1000) / (now - lastFpsT);
      frames = 0;
      lastFpsT = now;
      hud.textContent = `PELAGIA · GPU spike (1.0)\ncreatures ${n.toLocaleString()}\nFPS ${fps.toFixed(0)}`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
