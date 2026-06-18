import type { WorldConfig } from '../core/config.js';
import { Rng } from '../core/rng.js';
import { wrap, wrapDelta } from '../core/space.js';
import { FoodField } from './food.js';
import { SpatialGrid } from './grid.js';
import { Population } from './population.js';
import { INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, GENOME_SIZE, forward } from './brain.js';

const TWO_PI = Math.PI * 2;

/**
 * Dedicated RNG stream ids, so unrelated random processes (food placement,
 * creature initialisation, mutation) never share or perturb each other's
 * sequence. This keeps each subsystem independently reproducible.
 */
export const Stream = {
  Food: 1,
  CreatureInit: 2,
  Mutation: 3,
} as const;

/**
 * The simulation world: a toroidal 2D ocean of food and neurally-controlled
 * creatures. Each tick runs, in order:
 *
 *   1. spawn food
 *   2. (re)build the spatial grids
 *   3. sense -> think -> move  (double-buffered, like the GPU ping-pong: every
 *      creature reads tick-start state and writes to `next` buffers, applied at
 *      the end — so neighbour sensing is consistent and the CPU stays a faithful
 *      oracle for the double-buffered shader)
 *   4. eat + metabolise        (sequential by index; the order-sensitive part)
 *   5. reproduce               (original creatures above the energy threshold)
 *   6. die                     (energy <= 0, swap-removed)
 *
 * Everything advances deterministically from `config.seed`.
 */
export class World {
  readonly config: WorldConfig;
  readonly food: FoodField;
  readonly population: Population;
  tick = 0;
  /** Cumulative number of food pellets eaten (instrumentation / metrics / HUD). */
  foodConsumed = 0;

  private readonly foodRng: Rng;
  private readonly mutationRng: Rng;
  private readonly foodGrid: SpatialGrid;
  private readonly creatureGrid: SpatialGrid;
  private nextId: number;
  private foodSpawnAccumulator = 0;

  // Per-tick scratch (reused to avoid allocation).
  private readonly inputs = new Float32Array(INPUT_SIZE);
  private readonly hidden = new Float32Array(HIDDEN_SIZE);
  private readonly outputs = new Float32Array(OUTPUT_SIZE);
  private readonly nx: Float32Array;
  private readonly ny: Float32Array;
  private readonly nHeading: Float32Array;
  private readonly nSpeed: Float32Array;
  private readonly foodEaten: Uint8Array;

  constructor(config: WorldConfig) {
    this.config = config;
    this.food = new FoodField(config.foodCapacity);
    this.population = new Population(config.maxPopulation);
    this.foodRng = new Rng(config.seed, Stream.Food);
    this.mutationRng = new Rng(config.seed, Stream.Mutation);
    this.foodGrid = new SpatialGrid(config.width, config.height, config.perceptionRadius);
    this.creatureGrid = new SpatialGrid(config.width, config.height, config.perceptionRadius);
    this.nextId = config.initialPopulation;

    this.nx = new Float32Array(config.maxPopulation);
    this.ny = new Float32Array(config.maxPopulation);
    this.nHeading = new Float32Array(config.maxPopulation);
    this.nSpeed = new Float32Array(config.maxPopulation);
    this.foodEaten = new Uint8Array(config.foodCapacity);

    this.seedInitialFood();
    this.seedInitialPopulation();
  }

  /** Advance the world by one tick. */
  step(): void {
    const cfg = this.config;
    const pop = this.population;

    this.spawnFood();
    this.foodGrid.build(this.food.x, this.food.y, this.food.count);
    this.creatureGrid.build(pop.x, pop.y, pop.count);

    const n = pop.count; // creatures present at the start of this tick

    // --- Phase 1: sense + think + move (double-buffered) ---
    for (let i = 0; i < n; i++) {
      this.computeSensors(i);
      forward(pop.weights, i * GENOME_SIZE, this.inputs, this.hidden, this.outputs);

      const turn = this.outputs[0]!;
      const thrust = (this.outputs[1]! + 1) * 0.5; // [-1,1] -> [0,1]
      const heading = wrap(pop.heading[i]! + turn * cfg.maxTurnRate, TWO_PI);
      const speed = thrust * cfg.maxSpeed;

      this.nHeading[i] = heading;
      this.nSpeed[i] = speed;
      this.nx[i] = wrap(pop.x[i]! + Math.cos(heading) * speed, cfg.width);
      this.ny[i] = wrap(pop.y[i]! + Math.sin(heading) * speed, cfg.height);
    }
    for (let i = 0; i < n; i++) {
      pop.x[i] = this.nx[i]!;
      pop.y[i] = this.ny[i]!;
      pop.heading[i] = this.nHeading[i]!;
      pop.speed[i] = this.nSpeed[i]!;
    }

    // --- Phase 2: eat + metabolise ---
    this.foodEaten.fill(0, 0, this.food.count);
    for (let i = 0; i < n; i++) {
      const px = pop.x[i]!;
      const py = pop.y[i]!;
      let bestIdx = -1;
      let bestD2 = Infinity;
      this.foodGrid.forEachNeighbor(px, py, cfg.eatRadius, (idx, d2) => {
        if (!this.foodEaten[idx] && (d2 < bestD2 || (d2 === bestD2 && idx < bestIdx))) {
          bestD2 = d2;
          bestIdx = idx;
        }
      });
      let energy = pop.energy[i]!;
      if (bestIdx >= 0) {
        this.foodEaten[bestIdx] = 1;
        energy += cfg.foodEnergy;
        this.foodConsumed++;
      }
      energy -= cfg.baseCost + cfg.moveCost * pop.speed[i]!;
      pop.energy[i] = energy;
      pop.age[i] = pop.age[i]! + 1;
    }
    this.food.compact(this.foodEaten);

    // --- Phase 3: reproduction (original creatures only) ---
    for (let i = 0; i < n; i++) {
      if (pop.energy[i]! >= cfg.reproductionThreshold && pop.count < cfg.maxPopulation) {
        pop.reproduce(i, this.nextId++, this.mutationRng, cfg);
      }
    }

    // --- Phase 4: death (downward swap-remove keeps indices valid) ---
    for (let i = pop.count - 1; i >= 0; i--) {
      if (pop.energy[i]! <= 0) pop.removeSwap(i);
    }

    this.tick++;
  }

  /** Fill `this.inputs` with the egocentric sensor vector for creature `i`. */
  private computeSensors(i: number): void {
    const cfg = this.config;
    const pop = this.population;
    const px = pop.x[i]!;
    const py = pop.y[i]!;
    const ph = pop.heading[i]!;
    const inp = this.inputs;

    // Nearest food -> plankton channel. The CPU oracle has a single food type
    // (it is the Phase 0 forager), so the big-food channel (inp[3..5]) stays 0;
    // the GPU sim fills both channels. Determinism is unaffected — the oracle is
    // still fully reproducible and shares brain.ts/GENOME_SIZE with the GPU.
    const fIdx = this.foodGrid.findNearest(px, py, cfg.perceptionRadius);
    if (fIdx >= 0) {
      const dx = wrapDelta(this.food.x[fIdx]! - px, cfg.width);
      const dy = wrapDelta(this.food.y[fIdx]! - py, cfg.height);
      const dist = Math.sqrt(dx * dx + dy * dy);
      const rel = Math.atan2(dy, dx) - ph;
      inp[0] = Math.cos(rel);
      inp[1] = Math.sin(rel);
      inp[2] = Math.max(0, 1 - dist / cfg.perceptionRadius);
    } else {
      inp[0] = 0;
      inp[1] = 0;
      inp[2] = 0;
    }
    inp[3] = 0; // big-food channel (GPU-only)
    inp[4] = 0;
    inp[5] = 0;

    // Nearest other creature.
    let bestIdx = -1;
    let bestD2 = Infinity;
    this.creatureGrid.forEachNeighbor(px, py, cfg.perceptionRadius, (idx, d2) => {
      if (idx !== i && (d2 < bestD2 || (d2 === bestD2 && idx < bestIdx))) {
        bestD2 = d2;
        bestIdx = idx;
      }
    });
    if (bestIdx >= 0) {
      const dx = wrapDelta(pop.x[bestIdx]! - px, cfg.width);
      const dy = wrapDelta(pop.y[bestIdx]! - py, cfg.height);
      const dist = Math.sqrt(bestD2);
      const rel = Math.atan2(dy, dx) - ph;
      inp[6] = Math.cos(rel);
      inp[7] = Math.sin(rel);
      inp[8] = Math.max(0, 1 - dist / cfg.perceptionRadius);
    } else {
      inp[6] = 0;
      inp[7] = 0;
      inp[8] = 0;
    }

    inp[9] = pop.energy[i]! / cfg.reproductionThreshold;
    inp[10] = pop.speed[i]! / cfg.maxSpeed;
    inp[11] = 0; // temperature (GPU-only biome sense)
    inp[12] = 0; // school density (GPU-only sense)
    inp[13] = 0; // neighbour toxicity (GPU-only sense)
    inp[14] = 0; // neighbour relative size (GPU-only sense)
  }

  private seedInitialFood(): void {
    const target = Math.min(this.config.foodInitial, this.config.foodCapacity);
    for (let i = 0; i < target; i++) this.spawnOnePellet();
  }

  private seedInitialPopulation(): void {
    const cfg = this.config;
    const init = new Rng(cfg.seed, Stream.CreatureInit);
    const target = Math.min(cfg.initialPopulation, cfg.maxPopulation);
    for (let i = 0; i < target; i++) {
      const x = init.nextFloat() * cfg.width;
      const y = init.nextFloat() * cfg.height;
      const heading = init.nextRange(0, TWO_PI);
      const hue = init.nextFloat();
      this.population.spawnRandom(
        x,
        y,
        heading,
        cfg.initialEnergy,
        hue,
        i,
        init,
        cfg.weightInitStd,
      );
    }
  }

  private spawnFood(): void {
    this.foodSpawnAccumulator += this.config.foodSpawnPerTick;
    while (this.foodSpawnAccumulator >= 1 && this.food.count < this.config.foodCapacity) {
      this.spawnOnePellet();
      this.foodSpawnAccumulator -= 1;
    }
    if (this.food.count >= this.config.foodCapacity) this.foodSpawnAccumulator = 0;
  }

  private spawnOnePellet(): void {
    const x = this.foodRng.nextFloat() * this.config.width;
    const y = this.foodRng.nextFloat() * this.config.height;
    this.food.add(x, y);
  }
}
