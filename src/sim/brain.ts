import type { Rng } from '../core/rng.js';

/**
 * The creature brain: a fixed-topology multi-layer perceptron.
 *
 * Fixed topology (not NEAT) is a deliberate choice: every creature's network
 * has the same shape, so the genome is a flat weight vector and the forward pass
 * is identical across all creatures — which is exactly what lets thousands of
 * brains evaluate without thread divergence once this ports to a GPU compute
 * shader. The genome IS the weight vector; behaviour is never scripted.
 *
 * Architecture: INPUT_SIZE -> HIDDEN_SIZE (tanh) -> OUTPUT_SIZE (tanh).
 *
 * Sensor (input) layout, all egocentric and in roughly [-1, 1] (filled by the
 * simulation loop / GPU sense pass). Phase 6 splits the food sense into two
 * TYPE-SPECIFIC channels — nearest plankton and nearest big-food — so a brain
 * can steer toward its preferred resource even when the other type is closer.
 * That turns the two food types (D-023) into a real sensory niche (specialists),
 * not just a spatial one. The CPU oracle has a single food type, so it fills the
 * plankton channel from its nearest food and leaves the big-food channel at 0.
 *   0: planktonCos       cos of the angle to the nearest plankton, vs heading
 *   1: planktonSin       sin of that angle (plankton ahead / left / right)
 *   2: planktonProximity 1 at touching, 0 at/after perception radius
 *   3: bigFoodCos        same, for the nearest big-food pellet
 *   4: bigFoodSin
 *   5: bigFoodProximity
 *   6: neighborCos       same, for the nearest other creature
 *   7: neighborSin
 *   8: neighborProximity
 *   9: energyNorm        own energy, normalised
 *  10: speedNorm         own speed, normalised
 *
 * Output layout (each in [-1, 1]):
 *   0: turn          steer left/right, scaled to maxTurnRate
 *   1: thrustRaw     mapped to [0, 1] thrust by the loop
 */
export const INPUT_SIZE = 11;
export const HIDDEN_SIZE = 10;
export const OUTPUT_SIZE = 2;

/** Weights (including biases) per genome, excluding the activation genes. */
export const WEIGHT_GENES =
  INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE;

/**
 * Phase 6 — evolvable brain complexity. Each hidden neuron has an ACTIVATION GENE
 * appended after the weights (HIDDEN_SIZE of them). A neuron contributes only when
 * its gene is >= 0; mutation can flip a gene across zero, so offspring effectively
 * gain or lose neurons across generations — "growing new neurons", as in real
 * evolution — while the genome stays a fixed-size flat vector (GPU-friendly: every
 * brain shares one shape, no thread divergence; D-005). A zero gene means active,
 * so an all-zero genome behaves exactly as before.
 *
 * Phase 6 also appends ONE morphology gene — body size — after the activation
 * genes. Size is a phenotype gene (not used by the forward pass): the GPU sim
 * reads it for size-based predation, size-scaled metabolism/speed and the
 * on-screen body size. It is inherited and mutated like any other gene.
 */
export const GENOME_SIZE = WEIGHT_GENES + HIDDEN_SIZE + 1;

/** Index of the body-size gene within a genome (after the activation genes). */
export const SIZE_GENE = WEIGHT_GENES + HIDDEN_SIZE;
export const SIZE_MIN = 0.6;
export const SIZE_MAX = 2.2;

/** Map a raw size gene to a bounded body-size multiplier (gene 0 -> 1.0). */
export function sizeFromGene(gene: number): number {
  const s = 1 + 0.5 * gene;
  return s < SIZE_MIN ? SIZE_MIN : s > SIZE_MAX ? SIZE_MAX : s;
}

/**
 * Evaluate the network whose weights start at `offset` in `weights`.
 * `inputs` has length INPUT_SIZE; `hidden` and `outputs` are caller-owned
 * scratch buffers (reused across creatures to avoid per-tick allocation).
 *
 * Weight layout from `offset`: for each hidden unit, INPUT_SIZE input weights
 * then 1 bias; then for each output unit, HIDDEN_SIZE weights then 1 bias.
 */
export function forward(
  weights: Float32Array,
  offset: number,
  inputs: Float32Array,
  hidden: Float32Array,
  outputs: Float32Array,
): void {
  let p = offset;
  const act = offset + WEIGHT_GENES; // activation genes, one per hidden neuron

  for (let h = 0; h < HIDDEN_SIZE; h++) {
    let sum = 0;
    for (let i = 0; i < INPUT_SIZE; i++) sum += weights[p++]! * inputs[i]!;
    sum += weights[p++]!; // bias
    // Disabled neurons (activation gene < 0) contribute nothing this generation.
    hidden[h] = weights[act + h]! >= 0 ? Math.tanh(sum) : 0;
  }

  for (let o = 0; o < OUTPUT_SIZE; o++) {
    let sum = 0;
    for (let h = 0; h < HIDDEN_SIZE; h++) sum += weights[p++]! * hidden[h]!;
    sum += weights[p++]!; // bias
    outputs[o] = Math.tanh(sum);
  }
}

/** Fill one genome at `offset` with fresh random weights ~ N(0, std). */
export function randomGenome(weights: Float32Array, offset: number, rng: Rng, std: number): void {
  for (let i = 0; i < GENOME_SIZE; i++) weights[offset + i] = rng.nextGaussian() * std;
}

/**
 * Copy a parent genome to an offspring slot, mutating each weight with
 * probability `rate` by adding N(0, std). Deterministic given `rng`.
 */
export function mutateGenome(
  src: Float32Array,
  srcOffset: number,
  dst: Float32Array,
  dstOffset: number,
  rng: Rng,
  rate: number,
  std: number,
): void {
  for (let i = 0; i < GENOME_SIZE; i++) {
    let w = src[srcOffset + i]!;
    if (rng.nextFloat() < rate) w += rng.nextGaussian() * std;
    dst[dstOffset + i] = w;
  }
}
