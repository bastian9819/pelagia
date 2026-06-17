import { describe, it, expect } from 'vitest';
import { encodeHash, decodeHash, type ShareState } from '../src/gpu/share.js';

const sample: ShareState = {
  seed: 1234567890,
  n: 20000,
  params: [
    { idx: 15, value: 100 },
    { idx: 12, value: 0.1 },
    { idx: 3, value: 4 },
  ],
};

describe('encodeHash', () => {
  it('produces a compact, versioned fragment', () => {
    expect(encodeHash(sample)).toBe('#v=1&s=1234567890&n=20000&g=15:100,12:0.1,3:4');
  });

  it('omits the g= field when there are no god params', () => {
    expect(encodeHash({ seed: 7, n: 100, params: [] })).toBe('#v=1&s=7&n=100');
  });

  it('coerces the seed to unsigned 32-bit', () => {
    expect(encodeHash({ seed: -1, n: 1, params: [] })).toBe('#v=1&s=4294967295&n=1');
  });
});

describe('decodeHash', () => {
  it('round-trips a full state', () => {
    const parsed = decodeHash(encodeHash(sample));
    expect(parsed).toEqual(sample);
  });

  it('tolerates a missing leading #', () => {
    expect(decodeHash('s=42&n=5')).toEqual({ seed: 42, n: 5 });
  });

  it('returns null for empty or unusable hashes', () => {
    expect(decodeHash('')).toBeNull();
    expect(decodeHash('#')).toBeNull();
    expect(decodeHash('#nope=1')).toBeNull();
  });

  it('drops malformed or out-of-range god params but keeps valid ones', () => {
    const parsed = decodeHash('#s=1&n=1&g=15:100,99:5,7:abc,3:4');
    expect(parsed).toEqual({
      seed: 1,
      n: 1,
      params: [
        { idx: 15, value: 100 },
        { idx: 3, value: 4 },
      ],
    });
  });

  it('ignores a negative or zero N', () => {
    expect(decodeHash('#s=1&n=0')).toEqual({ seed: 1 });
  });

  it('rounds long fractional values to keep URLs short', () => {
    expect(encodeHash({ seed: 1, n: 1, params: [{ idx: 13, value: 0.123456789 }] })).toBe(
      '#v=1&s=1&n=1&g=13:0.1235',
    );
  });
});
