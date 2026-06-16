import { describe, it, expect } from 'vitest';
import { SpatialGrid } from '../src/sim/grid.js';
import { toroidalDistSq } from '../src/core/space.js';
import { Rng } from '../src/core/rng.js';

function bruteNeighbors(
  x: Float32Array,
  y: Float32Array,
  count: number,
  qx: number,
  qy: number,
  r: number,
  w: number,
  h: number,
): Set<number> {
  const out = new Set<number>();
  const r2 = r * r;
  for (let i = 0; i < count; i++) {
    if (toroidalDistSq(qx, qy, x[i]!, y[i]!, w, h) <= r2) out.add(i);
  }
  return out;
}

function bruteNearest(
  x: Float32Array,
  y: Float32Array,
  count: number,
  qx: number,
  qy: number,
  r: number,
  w: number,
  h: number,
): number {
  const r2 = r * r;
  let best = -1;
  let bestD2 = Infinity;
  for (let i = 0; i < count; i++) {
    const d2 = toroidalDistSq(qx, qy, x[i]!, y[i]!, w, h);
    if (d2 <= r2 && (d2 < bestD2 || (d2 === bestD2 && i < best))) {
      bestD2 = d2;
      best = i;
    }
  }
  return best;
}

describe('SpatialGrid vs brute force', () => {
  it('returns exactly the brute-force neighbour set across many configs', () => {
    const w = 200;
    const h = 200;
    for (const seed of [1, 2, 3, 7, 42]) {
      for (const cellSize of [10, 25, 50, 200]) {
        const rng = new Rng(seed);
        const count = 150;
        const x = new Float32Array(count);
        const y = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          x[i] = rng.nextFloat() * w;
          y[i] = rng.nextFloat() * h;
        }
        const grid = new SpatialGrid(w, h, cellSize);
        grid.build(x, y, count);

        for (const radius of [5, 15, 40, 90]) {
          for (let q = 0; q < 20; q++) {
            const qx = rng.nextFloat() * w;
            const qy = rng.nextFloat() * h;

            const expected = bruteNeighbors(x, y, count, qx, qy, radius, w, h);
            const got: number[] = [];
            grid.forEachNeighbor(qx, qy, radius, (idx) => got.push(idx));

            // No duplicate visits.
            expect(new Set(got).size).toBe(got.length);
            // Same membership.
            expect(new Set(got)).toEqual(expected);
            // Nearest matches.
            expect(grid.findNearest(qx, qy, radius)).toBe(
              bruteNearest(x, y, count, qx, qy, radius, w, h),
            );
          }
        }
      }
    }
  });

  it('finds neighbours across the toroidal seam', () => {
    const grid = new SpatialGrid(100, 100, 20);
    const x = new Float32Array([99, 1]);
    const y = new Float32Array([50, 50]);
    grid.build(x, y, 2);
    // Query at x=0: both points are within distance 1..2 across the seam.
    const got: number[] = [];
    grid.forEachNeighbor(0, 50, 3, (idx) => got.push(idx));
    expect(new Set(got)).toEqual(new Set([0, 1]));
  });

  it('returns -1 / nothing when empty', () => {
    const grid = new SpatialGrid(100, 100, 20);
    grid.build(new Float32Array(0), new Float32Array(0), 0);
    expect(grid.findNearest(50, 50, 50)).toBe(-1);
    let calls = 0;
    grid.forEachNeighbor(50, 50, 50, () => calls++);
    expect(calls).toBe(0);
  });

  it('breaks nearest ties toward the lower index', () => {
    const grid = new SpatialGrid(100, 100, 50);
    // Two points equidistant from the query at (50,50).
    const x = new Float32Array([40, 60]);
    const y = new Float32Array([50, 50]);
    grid.build(x, y, 2);
    expect(grid.findNearest(50, 50, 20)).toBe(0);
  });
});
