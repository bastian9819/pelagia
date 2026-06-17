import type { Rng } from '../core/rng.js';
import type { WorldConfig } from '../core/config.js';
import { wrap } from '../core/space.js';
import { GENOME_SIZE, randomGenome, mutateGenome } from './brain.js';

/** How much an offspring's lineage hue drifts from its parent's, per birth. */
const LINEAGE_HUE_DRIFT = 0.015;

/** Wrap a hue into [0, 1). */
function wrapHue(h: number): number {
  const m = h % 1;
  return m < 0 ? m + 1 : m;
}

/**
 * The creature population, stored as a structure-of-arrays packed in [0, count).
 *
 * Each creature carries: kinematic state (position, heading, speed), energy,
 * age, generation, a stable id (for identity / brain inspector / metrics), a
 * lineage hue (inherited with small drift so lineages are visually traceable),
 * and a slice of the flat genome buffer.
 *
 * Birth appends at `count`; death is an O(1) swap-remove. The SoA layout maps
 * directly onto GPU storage buffers for the Phase 1 port.
 */
export class Population {
  readonly capacity: number;
  count = 0;

  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly heading: Float32Array;
  readonly speed: Float32Array;
  readonly energy: Float32Array;
  readonly age: Uint32Array;
  readonly generation: Uint32Array;
  readonly hue: Float32Array;
  /** Stable per-creature id (stored as f64 to hold large counters exactly). */
  readonly id: Float64Array;
  /** Flat genome buffer: creature i's weights occupy [i*GENOME_SIZE, ...). */
  readonly weights: Float32Array;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.x = new Float32Array(capacity);
    this.y = new Float32Array(capacity);
    this.heading = new Float32Array(capacity);
    this.speed = new Float32Array(capacity);
    this.energy = new Float32Array(capacity);
    this.age = new Uint32Array(capacity);
    this.generation = new Uint32Array(capacity);
    this.hue = new Float32Array(capacity);
    this.id = new Float64Array(capacity);
    this.weights = new Float32Array(capacity * GENOME_SIZE);
  }

  /** Add a creature with a fresh random brain. Returns its index, or -1 if full. */
  spawnRandom(
    x: number,
    y: number,
    heading: number,
    energy: number,
    hue: number,
    id: number,
    rng: Rng,
    weightInitStd: number,
  ): number {
    if (this.count >= this.capacity) return -1;
    const i = this.count++;
    this.x[i] = x;
    this.y[i] = y;
    this.heading[i] = heading;
    this.speed[i] = 0;
    this.energy[i] = energy;
    this.age[i] = 0;
    this.generation[i] = 0;
    this.hue[i] = wrapHue(hue);
    this.id[i] = id;
    randomGenome(this.weights, i * GENOME_SIZE, rng, weightInitStd);
    return i;
  }

  /**
   * Spawn an offspring of the creature at `parent`. The parent passes a fraction
   * of its energy to the child; the child inherits a mutated genome and a hue
   * drifted slightly from the parent's. Offspring are born at the parent's
   * position (local reproduction -> spatial niches). Returns the child index, or
   * -1 if the population is full.
   *
   * NOTE: this draws from `rng` in caller-determined (index) order, which is
   * deterministic on CPU. The GPU port will re-key these draws by (id, tick) so
   * parallel reproduction stays reproducible.
   */
  reproduce(parent: number, childId: number, rng: Rng, config: WorldConfig): number {
    if (this.count >= this.capacity) return -1;
    const child = this.count++;

    const passed = this.energy[parent]! * config.offspringEnergyFraction;
    this.energy[parent]! -= passed;
    this.energy[child] = passed;

    // Tiny positional jitter so siblings don't perfectly overlap (wrapped, so a
    // birth near an edge stays inside the toroidal world).
    this.x[child] = wrap(this.x[parent]! + rng.nextRange(-2, 2), config.width);
    this.y[child] = wrap(this.y[parent]! + rng.nextRange(-2, 2), config.height);
    this.heading[child] = rng.nextRange(0, Math.PI * 2);
    this.speed[child] = 0;
    this.age[child] = 0;
    this.generation[child] = this.generation[parent]! + 1;
    this.hue[child] = wrapHue(this.hue[parent]! + rng.nextGaussian() * LINEAGE_HUE_DRIFT);
    this.id[child] = childId;

    mutateGenome(
      this.weights,
      parent * GENOME_SIZE,
      this.weights,
      child * GENOME_SIZE,
      rng,
      config.mutationRate,
      config.mutationStd,
    );
    return child;
  }

  /** Remove the creature at `index` by swapping the last one into its slot. */
  removeSwap(index: number): void {
    const last = this.count - 1;
    if (index < 0 || index > last) return;
    if (index !== last) {
      this.x[index] = this.x[last]!;
      this.y[index] = this.y[last]!;
      this.heading[index] = this.heading[last]!;
      this.speed[index] = this.speed[last]!;
      this.energy[index] = this.energy[last]!;
      this.age[index] = this.age[last]!;
      this.generation[index] = this.generation[last]!;
      this.hue[index] = this.hue[last]!;
      this.id[index] = this.id[last]!;
      this.weights.copyWithin(index * GENOME_SIZE, last * GENOME_SIZE, (last + 1) * GENOME_SIZE);
    }
    this.count--;
  }
}
