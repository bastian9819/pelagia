import type { WorldConfig } from '../core/config.js';
import { DEFAULT_CONFIG } from '../core/config.js';
import { Rng } from '../core/rng.js';
import { World } from './world.js';
import { GENOME_SIZE, randomGenome } from './brain.js';

/**
 * Emergence metric for the Phase 0 gate.
 *
 * The core question: do *evolved* brains forage better than *random* brains?
 * To answer it cleanly we use a controlled assay — both brain sets are dropped
 * into an identical arena (same seed => same start positions and food) with the
 * lifecycle frozen (no births/deaths), and we measure food collected per
 * creature. The only thing that differs between the two runs is the brains, so
 * any gap is foraging competence, not population dynamics or food-density luck.
 */

/** Stream id for generating control (random) brains; distinct from sim streams. */
const CONTROL_STREAM = 9001;

export interface GateOptions {
  /** Ticks of live evolution before extracting the evolved brains. */
  evolveTicks: number;
  /** Creatures used in each assay. */
  assayK: number;
  /** Ticks each assay runs. */
  assayWindow: number;
  /** Constant food count maintained in the assay arena (density). */
  assayFood: number;
  /** Assay arena seeds; results are averaged over them. */
  assaySeeds: number[];
  /** Minimum surviving population for the run to count as self-sustaining. */
  sustainMin: number;
}

export const DEFAULT_GATE_OPTIONS: GateOptions = {
  evolveTicks: 4000,
  assayK: 100,
  assayWindow: 400,
  assayFood: 150,
  assaySeeds: [101, 202, 303],
  sustainMin: 25,
};

export interface SeedResult {
  seed: number;
  evolvedRate: number;
  randomRate: number;
  ratio: number;
  evolvedCount: number;
  sustained: boolean;
  pass: boolean;
}

export interface GateReport {
  pass: boolean;
  passFraction: number;
  ratioThreshold: number;
  seedPassFraction: number;
  results: SeedResult[];
}

/** Assay config: frozen lifecycle, constant food density, training kinematics. */
function assayConfig(base: WorldConfig, k: number, seed: number, foodN: number): WorldConfig {
  return {
    ...base,
    seed,
    initialPopulation: k,
    maxPopulation: k,
    initialEnergy: 1e6, // never starve during the window
    reproductionThreshold: Number.POSITIVE_INFINITY, // never reproduce
    foodInitial: foodN,
    foodCapacity: foodN,
    foodSpawnPerTick: k, // refill aggressively so density stays ~constant
  };
}

/**
 * Drop `k` brains into an identical arena and return mean food eaten per
 * creature over the window, averaged across the assay seeds.
 */
export function assayForaging(
  brains: Float32Array,
  k: number,
  opts: GateOptions,
  base: WorldConfig = DEFAULT_CONFIG,
): number {
  let total = 0;
  for (const seed of opts.assaySeeds) {
    const w = new World(assayConfig(base, k, seed, opts.assayFood));
    w.population.weights.set(brains.subarray(0, k * GENOME_SIZE));
    for (let t = 0; t < opts.assayWindow; t++) w.step();
    total += w.foodConsumed / k;
  }
  return total / opts.assaySeeds.length;
}

/** Generate `k` fresh random control genomes. */
export function makeRandomBrains(k: number, seed: number, weightInitStd: number): Float32Array {
  const brains = new Float32Array(k * GENOME_SIZE);
  const rng = new Rng(seed, CONTROL_STREAM);
  for (let i = 0; i < k; i++) randomGenome(brains, i * GENOME_SIZE, rng, weightInitStd);
  return brains;
}

/** Evolve a world and measure the gate metric for one seed. */
export function measureGateForSeed(
  seed: number,
  opts: GateOptions = DEFAULT_GATE_OPTIONS,
  base: WorldConfig = DEFAULT_CONFIG,
): Omit<SeedResult, 'pass'> {
  const world = new World({ ...base, seed });
  for (let t = 0; t < opts.evolveTicks; t++) world.step();

  const evolvedCount = world.population.count;
  const k = Math.min(evolvedCount, opts.assayK);
  const sustained = evolvedCount >= opts.sustainMin;

  // If the population collapsed there is nothing meaningful to assay.
  if (k <= 0) {
    return { seed, evolvedRate: 0, randomRate: 0, ratio: 0, evolvedCount, sustained: false };
  }

  const evolvedBrains = world.population.weights.slice(0, k * GENOME_SIZE);
  const randomBrains = makeRandomBrains(k, seed, base.weightInitStd);

  const evolvedRate = assayForaging(evolvedBrains, k, opts, base);
  const randomRate = assayForaging(randomBrains, k, opts, base);
  const ratio = randomRate > 0 ? evolvedRate / randomRate : evolvedRate > 0 ? Infinity : 1;

  return { seed, evolvedRate, randomRate, ratio, evolvedCount, sustained };
}

/**
 * Evaluate the gate across many seeds. Passes when a seed's evolved foraging is
 * at least `ratioThreshold`x the random control AND the run was self-sustaining;
 * the overall gate passes when at least `seedPassFraction` of seeds pass.
 */
export function evaluateGate(
  seeds: number[],
  opts: GateOptions = DEFAULT_GATE_OPTIONS,
  ratioThreshold = 3,
  seedPassFraction = 0.75,
  base: WorldConfig = DEFAULT_CONFIG,
): GateReport {
  const results: SeedResult[] = seeds.map((seed) => {
    const r = measureGateForSeed(seed, opts, base);
    return { ...r, pass: r.sustained && r.ratio >= ratioThreshold };
  });
  const passFraction = results.filter((r) => r.pass).length / results.length;
  return {
    pass: passFraction >= seedPassFraction,
    passFraction,
    ratioThreshold,
    seedPassFraction,
    results,
  };
}
