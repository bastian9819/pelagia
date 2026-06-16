import { describe, it, expect } from 'vitest';
import { World } from '../src/sim/world.js';
import { DEFAULT_CONFIG, type WorldConfig } from '../src/core/config.js';

function makeConfig(overrides: Partial<WorldConfig> = {}): WorldConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

describe('World food economy', () => {
  it('seeds the initial food count', () => {
    const w = new World(makeConfig({ foodInitial: 800, foodCapacity: 1500 }));
    expect(w.food.count).toBe(800);
  });

  it('caps initial food at capacity', () => {
    const w = new World(makeConfig({ foodInitial: 2000, foodCapacity: 500 }));
    expect(w.food.count).toBe(500);
  });

  it('spawns food over time without exceeding capacity (no creatures eating)', () => {
    const w = new World(
      makeConfig({ initialPopulation: 0, foodInitial: 0, foodCapacity: 50, foodSpawnPerTick: 10 }),
    );
    expect(w.food.count).toBe(0);
    for (let i = 0; i < 100; i++) w.step();
    expect(w.food.count).toBe(50);
  });

  it('places all food within world bounds', () => {
    const cfg = makeConfig();
    const w = new World(cfg);
    for (let i = 0; i < 50; i++) w.step();
    for (let i = 0; i < w.food.count; i++) {
      expect(w.food.x[i]).toBeGreaterThanOrEqual(0);
      expect(w.food.x[i]).toBeLessThan(cfg.width);
      expect(w.food.y[i]).toBeGreaterThanOrEqual(0);
      expect(w.food.y[i]).toBeLessThan(cfg.height);
    }
  });

  it('advances the tick counter', () => {
    const w = new World(makeConfig());
    w.step();
    w.step();
    expect(w.tick).toBe(2);
  });
});

describe('World determinism', () => {
  it('produces byte-identical food layout for the same config', () => {
    const cfg = makeConfig();
    const a = new World(cfg);
    const b = new World(cfg);
    for (let i = 0; i < 200; i++) {
      a.step();
      b.step();
    }
    expect(a.food.count).toBe(b.food.count);
    for (let i = 0; i < a.food.count; i++) {
      expect(a.food.x[i]).toBe(b.food.x[i]);
      expect(a.food.y[i]).toBe(b.food.y[i]);
    }
  });

  it('produces a different layout for a different seed', () => {
    const a = new World(makeConfig({ seed: 1 }));
    const b = new World(makeConfig({ seed: 2 }));
    // First pellet position should differ.
    expect(a.food.x[0]).not.toBe(b.food.x[0]);
  });
});
