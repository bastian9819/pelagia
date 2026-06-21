/**
 * Likely cause of a lineage's extinction — a "coroner's verdict" inferred from
 * real signals: the clade's own evolved traits, how big it ever got, and the
 * world at the moment it died. It's a best-guess narrative (not a logged kill),
 * surfaced in the cladogram as "likely cause". Pure + testable so the mapping is
 * pinned down.
 */

/** The evolved character of a clade (from characterizeGenome) + its peak size. */
export interface CladeStats {
  /** Largest population it ever reached. */
  peak: number;
  /** Steer-toward-neighbour drive: <0 = flees (prey), >0 = hunts. */
  aggression: number;
  /** Thrust when food is dead ahead (0..1). */
  forage: number;
  /** Turn-toward-food strength. */
  seek: number;
  /** Headline behaviour label key (desc_*). */
  descKey: string;
}

/** The world at the time of death. */
export interface DeathContext {
  /** Predation gain (0 = predation disabled). */
  predGain: number;
  /** Total creatures alive. */
  aliveCount: number;
  /** Total food pellets available. */
  foodAlive: number;
}

export interface ExtinctionCause {
  /** i18n key (ext_*). */
  key: string;
  /** Predator lineage id for ext_predated, else 0. */
  detail: number;
}

/**
 * Infer the likely cause. Order matters: the most specific / dramatic explanation
 * that fits wins. `predator` is the apex predator lineage id alive at death (-1 if
 * none) — used only for predation deaths.
 */
export function inferExtinctionCause(
  s: CladeStats,
  ctx: DeathContext,
  predator: number,
): ExtinctionCause {
  if (s.peak <= 3) return { key: 'ext_stillborn', detail: 0 };
  // Prey-like clade dying while predators roam → most likely eaten.
  if (ctx.predGain > 0 && s.aggression < -0.05 && predator >= 0)
    return { key: 'ext_predated', detail: predator };
  // Couldn't reliably steer to / chase food.
  if (s.forage < 0.38 && s.seek < 0.18) return { key: 'ext_slowfeeder', detail: 0 };
  // World-wide food scarcity (low food per capita) → famine.
  if (ctx.aliveCount > 0 && ctx.foodAlive / ctx.aliveCount < 0.5)
    return { key: 'ext_starved', detail: 0 };
  // Undirected swimmers that never converged on foraging.
  if (s.descKey === 'desc_erratic' || s.descKey === 'desc_circler' || s.descKey === 'desc_away')
    return { key: 'ext_lost', detail: 0 };
  // A decent clade that simply lost the numbers game.
  return { key: 'ext_outcompeted', detail: 0 };
}
