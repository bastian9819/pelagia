import { describe, it, expect } from 'vitest';
import { DEFAULT_CONFIG } from '../src/core/config.js';

// Scaffold sanity test: confirms the test runner is wired and the default world
// config is internally coherent. Real simulation tests arrive in tasks 0.1+.
describe('DEFAULT_CONFIG', () => {
  it('has a positive, finite world size', () => {
    expect(DEFAULT_CONFIG.width).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.height).toBeGreaterThan(0);
    expect(Number.isFinite(DEFAULT_CONFIG.width)).toBe(true);
    expect(Number.isFinite(DEFAULT_CONFIG.height)).toBe(true);
  });

  it('starts with a non-empty population', () => {
    expect(DEFAULT_CONFIG.initialPopulation).toBeGreaterThan(0);
  });

  it('uses an integer seed', () => {
    expect(Number.isInteger(DEFAULT_CONFIG.seed)).toBe(true);
  });
});
