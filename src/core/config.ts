/**
 * Default world parameters for PELAGIA.
 *
 * These are the knobs the simulation reads. They are intentionally kept as a
 * single flat, readonly object so a full configuration can be serialized into a
 * shareable URL later (seed + params = same ocean). Values get calibrated during
 * the Phase 0 emergence work (task 0.9) and most become god-mode sliders later.
 */
export interface WorldConfig {
  /** Seed for the deterministic PRNG. Same seed + same params = same ocean. */
  readonly seed: number;
  /** World extent in simulation units (toroidal / wrap-around). */
  readonly width: number;
  readonly height: number;

  // --- Population ---
  /** Creatures present at world creation. */
  readonly initialPopulation: number;
  /** Hard cap on simultaneous creatures (buffer capacity / carrying ceiling). */
  readonly maxPopulation: number;
  /** Energy a creature starts life with. */
  readonly initialEnergy: number;

  // --- Food (the limited resource that drives competition / anti-gray-soup) ---
  /** Maximum number of food pellets that can exist at once (carrying capacity). */
  readonly foodCapacity: number;
  /** Food pellets present at world creation. */
  readonly foodInitial: number;
  /** Average new food pellets spawned per tick (fractional allowed). */
  readonly foodSpawnPerTick: number;
  /** Energy a creature gains from eating one pellet. */
  readonly foodEnergy: number;
  /** Distance within which a creature eats a pellet. */
  readonly eatRadius: number;

  // --- Perception & movement ---
  /** Sensor range: how far a creature can perceive food / neighbours. */
  readonly perceptionRadius: number;
  /** Maximum speed in units per tick. */
  readonly maxSpeed: number;
  /** Maximum heading change per tick, in radians. */
  readonly maxTurnRate: number;

  // --- Metabolism ---
  /** Energy spent each tick just being alive. */
  readonly baseCost: number;
  /** Energy spent per unit of distance moved. */
  readonly moveCost: number;

  // --- Reproduction & mutation ---
  /** Energy at/above which a creature reproduces. */
  readonly reproductionThreshold: number;
  /** Fraction of the parent's energy passed to the offspring. */
  readonly offspringEnergyFraction: number;
  /** Probability each weight mutates when copied to an offspring. */
  readonly mutationRate: number;
  /** Standard deviation of a mutation perturbation. */
  readonly mutationStd: number;
  /** Standard deviation of weights in a fresh random genome. */
  readonly weightInitStd: number;
}

export const DEFAULT_CONFIG: WorldConfig = {
  seed: 1,
  width: 1024,
  height: 1024,

  initialPopulation: 300,
  maxPopulation: 2000,
  initialEnergy: 50,

  foodCapacity: 1500,
  foodInitial: 800,
  foodSpawnPerTick: 3,
  foodEnergy: 12,
  eatRadius: 8,

  perceptionRadius: 120,
  maxSpeed: 4,
  maxTurnRate: 0.4,

  baseCost: 0.15,
  moveCost: 0.05,

  reproductionThreshold: 90,
  offspringEnergyFraction: 0.5,
  mutationRate: 0.1,
  mutationStd: 0.3,
  weightInitStd: 1.0,
};
