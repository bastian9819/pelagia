import { describe, it, expect } from 'vitest';
import { wrap, wrapDelta, toroidalDistSq } from '../src/core/space.js';

describe('wrap', () => {
  it('keeps values already in range', () => {
    expect(wrap(5, 10)).toBe(5);
    expect(wrap(0, 10)).toBe(0);
  });

  it('wraps values above size', () => {
    expect(wrap(12, 10)).toBe(2);
    expect(wrap(20, 10)).toBe(0);
  });

  it('wraps negative values into [0, size)', () => {
    expect(wrap(-1, 10)).toBe(9);
    expect(wrap(-11, 10)).toBe(9);
  });
});

describe('wrapDelta', () => {
  it('leaves short deltas unchanged', () => {
    expect(wrapDelta(2, 10)).toBe(2);
    expect(wrapDelta(-2, 10)).toBe(-2);
  });

  it('takes the short way around the seam', () => {
    // Going from 1 to 9 is +8 naively, but -2 across the seam.
    expect(wrapDelta(8, 10)).toBe(-2);
    expect(wrapDelta(-8, 10)).toBe(2);
  });
});

describe('toroidalDistSq', () => {
  it('measures straight-line distance away from the seam', () => {
    expect(toroidalDistSq(0, 0, 3, 4, 100, 100)).toBeCloseTo(25);
  });

  it('measures the shorter wrapped distance across the seam', () => {
    // x: 1 and 99 on a width-100 torus are 2 apart, not 98.
    expect(toroidalDistSq(1, 0, 99, 0, 100, 100)).toBeCloseTo(4);
  });
});
