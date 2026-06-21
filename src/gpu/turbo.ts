/**
 * Adaptive turbo step-cap controller.
 *
 * At high speed the loop runs several full sim pipelines per rendered frame, which
 * can tank the frame rate on a heavy ocean (e.g. healing a spot blooms the
 * population). This caps how many sim ticks we run per frame, feeding back from the
 * real frame time so the UI stays smooth.
 *
 * It's an AIMD controller (like TCP congestion control): back off
 * MULTIPLICATIVELY the instant a frame runs slow, then grow back ADDITIVELY and
 * slowly when there's headroom. The slow additive ramp plus a wide dead band make
 * it settle at a stable step count instead of oscillating between fast and
 * cratered frames (the failure mode of the previous multiplicative-grow version).
 */

/** Frame-time dead band (ms). Below LO we have headroom; above HI we're too slow. */
export const TURBO_LO_MS = 14; // ~71 fps — grow the cap
export const TURBO_HI_MS = 24; // ~42 fps — shrink the cap
export const TURBO_MAX_STEPS = 32;

/** One control step: given the last frame time, return the new step cap. */
export function nextStepCap(maxSteps: number, frameDt: number): number {
  // Ignore implausible deltas (first frame, tab backgrounded, a GC stall).
  if (!(frameDt > 0 && frameDt < 250)) return maxSteps;
  if (frameDt > TURBO_HI_MS) return Math.max(1, maxSteps * 0.8); // slow → back off now
  if (frameDt < TURBO_LO_MS) return Math.min(TURBO_MAX_STEPS, maxSteps + 0.25); // headroom → creep up
  return maxSteps; // inside the dead band → hold steady
}
