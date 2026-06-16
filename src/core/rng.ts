/**
 * Deterministic, counter-based pseudo-random number generation.
 *
 * Why counter-based instead of a classic stateful LCG/Mersenne generator?
 * PELAGIA's simulation must stay reproducible (same seed + params = same ocean)
 * AND eventually run massively in parallel on the GPU, where thousands of
 * creatures update at once. A counter-based generator derives each random value
 * purely from integer keys — e.g. (entityId, tick, stream, drawIndex) — with no
 * shared mutable state. That makes draws independent of evaluation order, which
 * is what lets the GPU port stay deterministic.
 *
 * All arithmetic is done in unsigned 32-bit integer space using `Math.imul` and
 * `>>> 0`, which exactly mirrors WGSL's `u32` wrapping semantics. The CPU
 * reference (this file) is therefore a faithful oracle for the future shader.
 *
 * The core mixer is the PCG output hash (Jarzynski, "Hash Functions for GPU
 * Rendering"). Golden values are pinned in the tests so the algorithm can never
 * change silently and break shared-URL reproducibility.
 */

const TWO_POW_32 = 4294967296; // 2^32

/** Coerce to unsigned 32-bit. */
function u32(x: number): number {
  return x >>> 0;
}

/**
 * PCG hash: maps a u32 to a well-distributed u32. Pure and stateless.
 */
export function pcgHash(input: number): number {
  const state = u32(Math.imul(u32(input), 747796405) + 2891336453);
  const word = u32(Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737));
  return u32((word >>> 22) ^ word);
}

/** Mix two u32 keys into one well-distributed u32 (order matters: a then b). */
export function hash2(a: number, b: number): number {
  return pcgHash(u32(pcgHash(u32(a)) ^ u32(b)));
}

/** Mix three u32 keys. */
export function hash3(a: number, b: number, c: number): number {
  return pcgHash(u32(hash2(a, b) ^ u32(c)));
}

/** Mix four u32 keys — the typical (entityId, tick, stream, drawIndex) key. */
export function hash4(a: number, b: number, c: number, d: number): number {
  return pcgHash(u32(hash3(a, b, c) ^ u32(d)));
}

/** Convert a u32 to a float in [0, 1). */
export function floatFromU32(value: number): number {
  return u32(value) / TWO_POW_32;
}

/**
 * A lightweight stateful stream built on the counter-based hash. Convenient for
 * sequential draws (e.g. world initialisation) while remaining fully
 * reproducible: it is just `hash2(key, counter)` with an incrementing counter.
 */
export class Rng {
  private readonly key: number;
  private counter: number;

  constructor(seed: number, stream = 0) {
    this.key = hash2(u32(seed), u32(stream));
    this.counter = 0;
  }

  /** Next raw unsigned 32-bit value. */
  nextU32(): number {
    const value = hash2(this.key, this.counter);
    this.counter = u32(this.counter + 1);
    return value;
  }

  /** Next float in [0, 1). */
  nextFloat(): number {
    return floatFromU32(this.nextU32());
  }

  /** Next float in [min, max). */
  nextRange(min: number, max: number): number {
    return min + (max - min) * this.nextFloat();
  }

  /** Next integer in [min, max) (max exclusive). */
  nextInt(min: number, max: number): number {
    return min + Math.floor(this.nextFloat() * (max - min));
  }

  /** Standard normal sample (mean 0, std 1) via the Box–Muller transform. */
  nextGaussian(): number {
    // Guard against log(0); the smallest u32/2^32 is 0, so clamp.
    let u1 = this.nextFloat();
    const u2 = this.nextFloat();
    if (u1 < 1e-12) u1 = 1e-12;
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Derive an independent child stream from this one. Used to give each spawned
   * creature its own reproducible random stream without coupling to siblings.
   */
  fork(stream: number): Rng {
    return new Rng(this.key, stream);
  }
}
