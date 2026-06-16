/**
 * Default world parameters for PELAGIA.
 *
 * These are the knobs the simulation reads. They are intentionally kept as a
 * single flat, readonly object so a full configuration can be serialized into a
 * shareable URL later (seed + params = same ocean). Values get calibrated during
 * the Phase 0 emergence work (task 0.9).
 */
export interface WorldConfig {
  /** Seed for the deterministic PRNG. Same seed + same params = same ocean. */
  readonly seed: number;
  /** World extent in simulation units (toroidal / wrap-around). */
  readonly width: number;
  readonly height: number;

  /** Target starting population of creatures. */
  readonly initialPopulation: number;

  // --- Food (the limited resource that drives competition / anti-gray-soup) ---
  /** Maximum number of food pellets that can exist at once (carrying capacity). */
  readonly foodCapacity: number;
  /** Food pellets present at world creation. */
  readonly foodInitial: number;
  /** Average new food pellets spawned per tick (fractional allowed). */
  readonly foodSpawnPerTick: number;
  /** Energy a creature gains from eating one pellet. */
  readonly foodEnergy: number;
}

export const DEFAULT_CONFIG: WorldConfig = {
  seed: 1,
  width: 1024,
  height: 1024,
  initialPopulation: 300,

  foodCapacity: 1500,
  foodInitial: 800,
  foodSpawnPerTick: 3,
  foodEnergy: 12,
};
