/**
 * Default world parameters for PELAGIA.
 *
 * These are the knobs the simulation reads. They are intentionally kept as a
 * single plain object so a full configuration can be serialized into a shareable
 * URL later (seed + params = same ocean). Values here are placeholders for the
 * scaffold; they get calibrated during the Phase 0 emergence work (task 0.9).
 */
export interface WorldConfig {
  /** Seed for the deterministic PRNG. Same seed + same params = same ocean. */
  readonly seed: number;
  /** World extent in simulation units (toroidal / wrap-around). */
  readonly width: number;
  readonly height: number;
  /** Target starting population of creatures. */
  readonly initialPopulation: number;
}

export const DEFAULT_CONFIG: WorldConfig = {
  seed: 1,
  width: 1024,
  height: 1024,
  initialPopulation: 300,
};
