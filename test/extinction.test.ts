import { describe, it, expect } from 'vitest';
import { inferExtinctionCause, type CladeStats, type DeathContext } from '../src/gpu/extinction.js';

// A healthy-ish baseline clade + a comfortable world; tweak per case.
const clade = (o: Partial<CladeStats> = {}): CladeStats => ({
  peak: 200,
  aggression: 0,
  forage: 0.6,
  seek: 0.4,
  descKey: 'desc_chase',
  ...o,
});
const world = (o: Partial<DeathContext> = {}): DeathContext => ({
  predGain: 0,
  aliveCount: 8000,
  foodAlive: 8000,
  ...o,
});

describe('inferExtinctionCause', () => {
  it('stillborn when it never grew', () => {
    expect(inferExtinctionCause(clade({ peak: 2 }), world(), 5).key).toBe('ext_stillborn');
  });

  it('predated when a prey-like clade dies with predators active', () => {
    const c = inferExtinctionCause(clade({ aggression: -0.4 }), world({ predGain: 0.5 }), 4242);
    expect(c.key).toBe('ext_predated');
    expect(c.detail).toBe(4242); // names the apex predator
  });

  it('does NOT blame predation when predation is off or no predator known', () => {
    expect(
      inferExtinctionCause(clade({ aggression: -0.4 }), world({ predGain: 0 }), 7).key,
    ).not.toBe('ext_predated');
    expect(
      inferExtinctionCause(clade({ aggression: -0.4 }), world({ predGain: 0.5 }), -1).key,
    ).not.toBe('ext_predated');
  });

  it('slow feeder when it could barely steer to / chase food', () => {
    expect(inferExtinctionCause(clade({ forage: 0.2, seek: 0.05 }), world(), -1).key).toBe(
      'ext_slowfeeder',
    );
  });

  it('starved when food per capita collapsed', () => {
    expect(
      inferExtinctionCause(clade(), world({ foodAlive: 1000, aliveCount: 9000 }), -1).key,
    ).toBe('ext_starved');
  });

  it('lost when its strategy never converged on foraging', () => {
    expect(inferExtinctionCause(clade({ descKey: 'desc_circler' }), world(), -1).key).toBe(
      'ext_lost',
    );
  });

  it('outcompeted as the fallback for a capable clade that lost the numbers game', () => {
    expect(inferExtinctionCause(clade(), world(), -1).key).toBe('ext_outcompeted');
  });

  it('predation takes priority over being a slow feeder', () => {
    const c = inferExtinctionCause(
      clade({ aggression: -0.3, forage: 0.2, seek: 0.05 }),
      world({ predGain: 0.5 }),
      99,
    );
    expect(c.key).toBe('ext_predated');
  });
});
