import type { World } from './world.js';
import { GENOME_SIZE } from './brain.js';

/**
 * A 32-bit FNV-1a checksum of the full world state, hashing the exact float
 * bits. Used as a golden determinism oracle in tests (same seed + N steps =>
 * same hash) and handy as a state checksum when debugging or sharing an ocean.
 */
export function hashWorld(world: World): number {
  const fbits = new Float32Array(1);
  const fview = new Uint32Array(fbits.buffer);
  const dv = new DataView(new ArrayBuffer(8));

  let h = 2166136261 >>> 0;
  const mixU32 = (u: number): void => {
    h = (h ^ (u >>> 0)) >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  };
  const mixF32 = (x: number): void => {
    fbits[0] = x;
    mixU32(fview[0]!);
  };
  const mixF64 = (x: number): void => {
    dv.setFloat64(0, x);
    mixU32(dv.getUint32(0));
    mixU32(dv.getUint32(4));
  };

  mixU32(world.tick);

  const pop = world.population;
  const n = pop.count;
  mixU32(n);
  for (let i = 0; i < n; i++) {
    mixF32(pop.x[i]!);
    mixF32(pop.y[i]!);
    mixF32(pop.heading[i]!);
    mixF32(pop.speed[i]!);
    mixF32(pop.energy[i]!);
    mixF32(pop.hue[i]!);
    mixU32(pop.age[i]!);
    mixU32(pop.generation[i]!);
    mixF64(pop.id[i]!);
  }
  for (let i = 0; i < n * GENOME_SIZE; i++) mixF32(pop.weights[i]!);

  const food = world.food;
  mixU32(food.count);
  for (let i = 0; i < food.count; i++) {
    mixF32(food.x[i]!);
    mixF32(food.y[i]!);
  }

  return h >>> 0;
}
