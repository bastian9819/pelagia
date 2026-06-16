import { describe, it, expect } from 'vitest';
import {
  evaluateGate,
  measureGateForSeed,
  DEFAULT_GATE_OPTIONS,
  type GateOptions,
} from '../src/sim/metrics.js';

// Reduced parameters so this runs in a couple of seconds while still exercising
// real evolution. The full gate (evolveTicks 4000, 8 seeds, 3 assay arenas)
// scores 7.6x-14x; even reduced it clears the 3x bar by a wide margin, so the
// assertion is not flaky. This test locks the project's central property: that
// adaptive food-seeking actually emerges.
const FAST: GateOptions = {
  ...DEFAULT_GATE_OPTIONS,
  evolveTicks: 1200,
  assayWindow: 200,
  assaySeeds: [101],
};

describe('emergence gate', () => {
  it('evolved brains forage at least 3x better than random, sustainably', () => {
    const report = evaluateGate([1, 2, 3], FAST);
    expect(report.pass).toBe(true);
    for (const r of report.results) {
      expect(r.sustained).toBe(true);
      expect(r.ratio).toBeGreaterThanOrEqual(3);
      // Sanity: the evolved population actually out-forages, not a divide-by-zero.
      expect(r.evolvedRate).toBeGreaterThan(r.randomRate);
    }
  });

  it('is reproducible (same seed -> same measurement)', () => {
    const a = measureGateForSeed(1, FAST);
    const b = measureGateForSeed(1, FAST);
    expect(a.evolvedRate).toBe(b.evolvedRate);
    expect(a.randomRate).toBe(b.randomRate);
    expect(a.ratio).toBe(b.ratio);
    expect(a.evolvedCount).toBe(b.evolvedCount);
  });
});
