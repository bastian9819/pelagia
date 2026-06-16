import type { WorldConfig } from '../core/config.js';
import { Rng } from '../core/rng.js';
import { FoodField } from './food.js';

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
 * The simulation world: a toroidal 2D ocean holding food (and, from task 0.5,
 * creatures). Phase 0.2 implements the world substrate and the food economy;
 * the full perceive -> think -> act -> metabolise -> reproduce/die loop is wired
 * up in task 0.6.
 *
 * Everything advances deterministically from `config.seed`, so two worlds with
 * the same config produce byte-identical state after the same number of steps.
 */
export class World {
  readonly config: WorldConfig;
  readonly food: FoodField;
  tick = 0;

  private readonly foodRng: Rng;
  private foodSpawnAccumulator = 0;

  constructor(config: WorldConfig) {
    this.config = config;
    this.food = new FoodField(config.foodCapacity);
    this.foodRng = new Rng(config.seed, Stream.Food);
    this.seedInitialFood();
  }

  /** Advance the world by one tick. */
  step(): void {
    this.spawnFood();
    this.tick++;
  }

  private seedInitialFood(): void {
    const target = Math.min(this.config.foodInitial, this.config.foodCapacity);
    for (let i = 0; i < target; i++) this.spawnOnePellet();
  }

  private spawnFood(): void {
    this.foodSpawnAccumulator += this.config.foodSpawnPerTick;
    while (this.foodSpawnAccumulator >= 1 && this.food.count < this.config.foodCapacity) {
      this.spawnOnePellet();
      this.foodSpawnAccumulator -= 1;
    }
    // Don't let the accumulator build up unbounded while the field is full.
    if (this.food.count >= this.config.foodCapacity) this.foodSpawnAccumulator = 0;
  }

  private spawnOnePellet(): void {
    const x = this.foodRng.nextFloat() * this.config.width;
    const y = this.foodRng.nextFloat() * this.config.height;
    this.food.add(x, y);
  }
}
