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
 * simulation loop in task 0.6):
 *   0: foodCos       cos of the angle to the nearest food, relative to heading
 *   1: foodSin       sin of that angle (food ahead / left / right)
 *   2: foodProximity 1 at touching, 0 at/after perception radius
 *   3: neighborCos   same, for the nearest other creature
 *   4: neighborSin
 *   5: neighborProximity
 *   6: energyNorm    own energy, normalised
 *   7: speedNorm     own speed, normalised
 *
 * Output layout (each in [-1, 1]):
 *   0: turn          steer left/right, scaled to maxTurnRate
 *   1: thrustRaw     mapped to [0, 1] thrust by the loop
 */
export const INPUT_SIZE = 8;
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
 */
export const GENOME_SIZE = WEIGHT_GENES + HIDDEN_SIZE;

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
