import { describe, it, expect } from 'vitest';
import { pcgHash, hash2, hash4, floatFromU32, Rng } from '../src/core/rng.js';

describe('pcgHash', () => {
  it('is a pure function (same input -> same output)', () => {
    expect(pcgHash(12345)).toBe(pcgHash(12345));
  });

  it('matches pinned golden values (locks the algorithm)', () => {
    // If these ever change, shared-URL reproducibility breaks. Change on purpose only.
    expect(pcgHash(0)).toBe(129708002);
    expect(pcgHash(1)).toBe(2831084092);
    expect(hash2(1, 2)).toBe(3752912895);
    expect(hash4(7, 42, 3, 0)).toBe(868663099);
  });

  it('returns values in unsigned 32-bit range', () => {
    for (let i = 0; i < 1000; i++) {
      const v = pcgHash(i);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(2 ** 32);
      expect(Number.isInteger(v)).toBe(true);
    }
  });

  it('handles negative inputs as their u32 reinterpretation', () => {
    expect(pcgHash(-1)).toBe(pcgHash(0xffffffff));
  });
});

describe('floatFromU32', () => {
  it('maps to [0, 1)', () => {
    expect(floatFromU32(0)).toBe(0);
    expect(floatFromU32(0xffffffff)).toBeLessThan(1);
    expect(floatFromU32(0xffffffff)).toBeGreaterThan(0.999);
  });
});

describe('Rng', () => {
  it('is fully reproducible from the same seed', () => {
    const a = new Rng(1);
    const b = new Rng(1);
    const seqA = Array.from({ length: 50 }, () => a.nextU32());
    const seqB = Array.from({ length: 50 }, () => b.nextU32());
    expect(seqA).toEqual(seqB);
  });

  it('matches pinned golden sequence for seed 1', () => {
    const r = new Rng(1);
    expect([r.nextU32(), r.nextU32(), r.nextU32()]).toEqual([555269243, 2157230610, 1272754567]);
  });

  it('produces different sequences for different seeds', () => {
    const a = new Rng(1);
    const b = new Rng(2);
    expect(a.nextU32()).not.toBe(b.nextU32());
  });

  it('produces different sequences for different streams', () => {
    const a = new Rng(1, 0);
    const b = new Rng(1, 1);
    expect(a.nextU32()).not.toBe(b.nextU32());
  });

  it('nextFloat stays in [0, 1)', () => {
    const r = new Rng(99);
    for (let i = 0; i < 10000; i++) {
      const f = r.nextFloat();
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThan(1);
    }
  });

  it('nextFloat is approximately uniform (mean ~0.5)', () => {
    const r = new Rng(7);
    const n = 100000;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += r.nextFloat();
    expect(sum / n).toBeCloseTo(0.5, 2);
  });

  it('nextRange respects bounds', () => {
    const r = new Rng(3);
    for (let i = 0; i < 1000; i++) {
      const v = r.nextRange(-5, 10);
      expect(v).toBeGreaterThanOrEqual(-5);
      expect(v).toBeLessThan(10);
    }
  });

  it('nextInt respects bounds and is integral', () => {
    const r = new Rng(3);
    for (let i = 0; i < 1000; i++) {
      const v = r.nextInt(0, 7);
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(7);
    }
  });

  it('nextGaussian has ~0 mean and ~1 std', () => {
    const r = new Rng(11);
    const n = 100000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const g = r.nextGaussian();
      sum += g;
      sumSq += g * g;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(mean).toBeCloseTo(0, 1);
    expect(Math.sqrt(variance)).toBeCloseTo(1, 1);
  });

  it('fork produces an independent, reproducible stream', () => {
    const parent = new Rng(42);
    const childA = parent.fork(5);
    const childB = parent.fork(5);
    expect(childA.nextU32()).toBe(childB.nextU32());
    expect(parent.fork(5).nextU32()).not.toBe(parent.fork(6).nextU32());
  });
});
