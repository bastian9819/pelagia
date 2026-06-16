import { describe, it, expect } from 'vitest';
import { Population } from '../src/sim/population.js';
import { GENOME_SIZE } from '../src/sim/brain.js';
import { Rng } from '../src/core/rng.js';
import { DEFAULT_CONFIG } from '../src/core/config.js';

function spawnOne(pop: Population, id: number, rng: Rng, energy = 50): number {
  return pop.spawnRandom(100, 100, 0, energy, 0.5, id, rng, 1);
}

describe('Population spawn', () => {
  it('adds creatures and sets fields', () => {
    const pop = new Population(10);
    const i = spawnOne(pop, 7, new Rng(1), 42);
    expect(i).toBe(0);
    expect(pop.count).toBe(1);
    expect(pop.energy[0]).toBe(42);
    expect(pop.id[0]).toBe(7);
    expect(pop.generation[0]).toBe(0);
  });

  it('respects capacity', () => {
    const pop = new Population(2);
    const rng = new Rng(1);
    expect(spawnOne(pop, 0, rng)).toBe(0);
    expect(spawnOne(pop, 1, rng)).toBe(1);
    expect(spawnOne(pop, 2, rng)).toBe(-1);
    expect(pop.count).toBe(2);
  });
});

describe('Population reproduce', () => {
  it('splits energy and advances generation', () => {
    const pop = new Population(10);
    spawnOne(pop, 0, new Rng(1), 80);
    const child = pop.reproduce(0, 99, new Rng(2), DEFAULT_CONFIG);
    expect(child).toBe(1);
    // offspringEnergyFraction default 0.5 -> even split.
    expect(pop.energy[0]).toBeCloseTo(40);
    expect(pop.energy[1]).toBeCloseTo(40);
    expect(pop.generation[1]).toBe(1);
    expect(pop.id[1]).toBe(99);
  });

  it('is born near the parent with a similar hue', () => {
    const pop = new Population(10);
    pop.spawnRandom(500, 300, 0, 80, 0.7, 0, new Rng(1), 1);
    pop.reproduce(0, 1, new Rng(2), DEFAULT_CONFIG);
    expect(Math.abs(pop.x[1]! - 500)).toBeLessThanOrEqual(2);
    expect(Math.abs(pop.y[1]! - 300)).toBeLessThanOrEqual(2);
    expect(Math.abs(pop.hue[1]! - 0.7)).toBeLessThan(0.1);
  });

  it('mutates the genome (similar but not identical)', () => {
    const pop = new Population(10);
    spawnOne(pop, 0, new Rng(1), 80);
    pop.reproduce(0, 1, new Rng(2), DEFAULT_CONFIG);
    let diffs = 0;
    for (let i = 0; i < GENOME_SIZE; i++) {
      if (pop.weights[i] !== pop.weights[GENOME_SIZE + i]) diffs++;
    }
    expect(diffs).toBeGreaterThan(0);
    expect(diffs).toBeLessThan(GENOME_SIZE); // not everything changes at rate 0.1
  });

  it('respects capacity', () => {
    const pop = new Population(1);
    spawnOne(pop, 0, new Rng(1), 80);
    expect(pop.reproduce(0, 1, new Rng(2), DEFAULT_CONFIG)).toBe(-1);
  });
});

describe('Population removeSwap', () => {
  it('moves the last creature into the removed slot, genome included', () => {
    const pop = new Population(10);
    const rng = new Rng(1);
    spawnOne(pop, 10, rng);
    spawnOne(pop, 11, rng);
    spawnOne(pop, 12, rng);
    // Mark the last creature's first weight so we can track it.
    pop.weights[2 * GENOME_SIZE] = 123.5;

    pop.removeSwap(0);
    expect(pop.count).toBe(2);
    expect(pop.id[0]).toBe(12); // former last moved into slot 0
    expect(pop.weights[0]).toBe(123.5);
  });

  it('handles removing the last element', () => {
    const pop = new Population(10);
    const rng = new Rng(1);
    spawnOne(pop, 10, rng);
    spawnOne(pop, 11, rng);
    pop.removeSwap(1);
    expect(pop.count).toBe(1);
    expect(pop.id[0]).toBe(10);
  });
});

describe('Population determinism', () => {
  it('produces identical state from the same seeds', () => {
    function build(): Population {
      const pop = new Population(10);
      const init = new Rng(5);
      const mut = new Rng(6);
      for (let k = 0; k < 4; k++) {
        pop.spawnRandom(init.nextFloat() * 100, init.nextFloat() * 100, 0, 80, 0.5, k, init, 1);
      }
      pop.reproduce(0, 100, mut, DEFAULT_CONFIG);
      pop.reproduce(2, 101, mut, DEFAULT_CONFIG);
      return pop;
    }
    const a = build();
    const b = build();
    expect(a.count).toBe(b.count);
    expect(Array.from(a.energy.subarray(0, a.count))).toEqual(
      Array.from(b.energy.subarray(0, b.count)),
    );
    expect(Array.from(a.weights.subarray(0, a.count * GENOME_SIZE))).toEqual(
      Array.from(b.weights.subarray(0, b.count * GENOME_SIZE)),
    );
  });
});
