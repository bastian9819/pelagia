/** WebGPU device initialisation and capability detection. */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** Preferred swap-chain format for canvas configuration. */
  format: GPUTextureFormat;
}

export class WebGpuUnsupportedError extends Error {
  constructor(reason: string) {
    super(`WebGPU unavailable: ${reason}`);
    this.name = 'WebGpuUnsupportedError';
  }
}

export async function initGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) throw new WebGpuUnsupportedError('navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGpuUnsupportedError('no adapter');
  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();
  return { adapter, device, format };
}
