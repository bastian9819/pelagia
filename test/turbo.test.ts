import { describe, it, expect } from 'vitest';
import { nextStepCap, TURBO_LO_MS, TURBO_HI_MS, TURBO_MAX_STEPS } from '../src/gpu/turbo.js';

/**
 * Close-loop model of one rendered frame: we run `min(requestedSpeed, floor(cap))`
 * sim ticks, and the frame takes `base + steps*perStep` ms. This lets us assert the
 * controller SETTLES (no oscillation) under a heavy ocean — the founder's case of
 * healing blooming the population so each tick is expensive.
 */
function simulate(opts: {
  perStep: number; // ms of GPU work per sim tick (high = heavy ocean)
  base: number; // ms of fixed per-frame cost (render etc.)
  speed: number; // requested turbo multiplier
  frames: number;
}): { caps: number[]; dts: number[] } {
  let cap = 16;
  const caps: number[] = [];
  const dts: number[] = [];
  for (let i = 0; i < opts.frames; i++) {
    const steps = Math.max(1, Math.min(opts.speed, Math.floor(cap)));
    const frameDt = opts.base + steps * opts.perStep;
    cap = nextStepCap(cap, frameDt);
    caps.push(cap);
    dts.push(frameDt);
  }
  return { caps, dts };
}

describe('nextStepCap (adaptive turbo controller)', () => {
  it('backs off immediately on a slow frame and never below 1', () => {
    expect(nextStepCap(16, 50)).toBeLessThan(16);
    expect(nextStepCap(1, 50)).toBe(1); // floor
    expect(nextStepCap(1.2, 100)).toBe(1);
  });

  it('grows only additively (slowly) when there is headroom', () => {
    const grown = nextStepCap(8, 5);
    expect(grown).toBeGreaterThan(8);
    expect(grown - 8).toBeLessThanOrEqual(0.25); // gentle, not multiplicative
    expect(nextStepCap(TURBO_MAX_STEPS, 5)).toBe(TURBO_MAX_STEPS); // capped
  });

  it('holds steady inside the dead band', () => {
    const mid = (TURBO_LO_MS + TURBO_HI_MS) / 2;
    expect(nextStepCap(7.3, mid)).toBe(7.3);
  });

  it('ignores implausible frame deltas (first frame / backgrounded tab)', () => {
    expect(nextStepCap(10, 0)).toBe(10);
    expect(nextStepCap(10, 999)).toBe(10);
  });

  it('settles without oscillating on a heavy ocean at high turbo', () => {
    // perStep 6ms (heavy, dense bloom), 8x requested: 8 ticks would be ~49ms (~20fps).
    const { caps, dts } = simulate({ perStep: 6, base: 1, speed: 8, frames: 400 });
    const tailCaps = caps.slice(-60);
    const tailDts = dts.slice(-60);
    // Frame time stays in a smooth range — no cratering to single digits.
    const worst = Math.max(...tailDts);
    expect(worst).toBeLessThan(28); // > ~36fps at all times once settled
    // The effective tick count is stable (small spread = no oscillation).
    const usedSteps = tailCaps.map((c) => Math.min(8, Math.floor(c)));
    const spread = Math.max(...usedSteps) - Math.min(...usedSteps);
    expect(spread).toBeLessThanOrEqual(1);
  });

  it('uses the full requested speed when the ocean is light', () => {
    // perStep 1ms, 4x requested: 4 ticks = ~5ms, tons of headroom → cap climbs past 4.
    const { caps } = simulate({ perStep: 1, base: 1, speed: 4, frames: 200 });
    expect(Math.floor(caps[caps.length - 1]!)).toBeGreaterThanOrEqual(4);
  });
});
