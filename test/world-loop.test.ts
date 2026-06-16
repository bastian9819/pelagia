import { describe, it, expect } from 'vitest';
import { World } from '../src/sim/world.js';
import { hashWorld } from '../src/sim/hash.js';
import { DEFAULT_CONFIG, type WorldConfig } from '../src/core/config.js';

function makeConfig(overrides: Partial<WorldConfig> = {}): WorldConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

describe('integrated simulation loop', () => {
  it('runs many ticks without error and keeps invariants', () => {
    const cfg = makeConfig();
    const w = new World(cfg);
    for (let t = 0; t < 300; t++) {
      w.step();
      const pop = w.population;
      expect(pop.count).toBeGreaterThanOrEqual(0);
      expect(pop.count).toBeLessThanOrEqual(cfg.maxPopulation);
      expect(w.food.count).toBeLessThanOrEqual(cfg.foodCapacity);
      for (let i = 0; i < pop.count; i++) {
        // Dead creatures are removed each tick, so survivors have energy > 0.
        expect(pop.energy[i]!).toBeGreaterThan(0);
        // Positions stay wrapped inside the toroidal world.
        expect(pop.x[i]!).toBeGreaterThanOrEqual(0);
        expect(pop.x[i]!).toBeLessThan(cfg.width);
        expect(pop.y[i]!).toBeGreaterThanOrEqual(0);
        expect(pop.y[i]!).toBeLessThan(cfg.height);
      }
    }
  });

  it('is bit-for-bit deterministic across runs (golden state hash)', () => {
    const a = new World(makeConfig());
    const b = new World(makeConfig());
    for (let t = 1; t <= 250; t++) {
      a.step();
      b.step();
      if (t === 50 || t === 150 || t === 250) {
        expect(hashWorld(a)).toBe(hashWorld(b));
      }
    }
  });

  it('produces different histories for different seeds', () => {
    const a = new World(makeConfig({ seed: 1 }));
    const b = new World(makeConfig({ seed: 2 }));
    for (let t = 0; t < 100; t++) {
      a.step();
      b.step();
    }
    expect(hashWorld(a)).not.toBe(hashWorld(b));
  });

  it('lets creatures reproduce (generations advance beyond the founders)', () => {
    const w = new World(makeConfig());
    for (let t = 0; t < 500; t++) w.step();
    let maxGen = 0;
    for (let i = 0; i < w.population.count; i++) {
      if (w.population.generation[i]! > maxGen) maxGen = w.population.generation[i]!;
    }
    // Even with random brains, some lineages should manage to reproduce.
    expect(maxGen).toBeGreaterThan(0);
  });
});
