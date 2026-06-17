import { describe, it, expect } from 'vitest';
import {
  INPUT_SIZE,
  HIDDEN_SIZE,
  OUTPUT_SIZE,
  WEIGHT_GENES,
  GENOME_SIZE,
  forward,
  randomGenome,
  mutateGenome,
} from '../src/sim/brain.js';
import { Rng } from '../src/core/rng.js';

function scratch() {
  return {
    inputs: new Float32Array(INPUT_SIZE),
    hidden: new Float32Array(HIDDEN_SIZE),
    outputs: new Float32Array(OUTPUT_SIZE),
  };
}

describe('brain architecture', () => {
  it('has the expected genome size (weights + one activation gene per hidden unit)', () => {
    expect(WEIGHT_GENES).toBe(
      INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE,
    );
    expect(WEIGHT_GENES).toBe(112);
    expect(GENOME_SIZE).toBe(WEIGHT_GENES + HIDDEN_SIZE);
    expect(GENOME_SIZE).toBe(122);
  });
});

describe('forward', () => {
  it('outputs zero for an all-zero genome (tanh(0) = 0)', () => {
    const w = new Float32Array(GENOME_SIZE);
    const s = scratch();
    s.inputs.fill(0.7);
    forward(w, 0, s.inputs, s.hidden, s.outputs);
    expect(Array.from(s.outputs)).toEqual([0, 0]);
  });

  it('matches the hand-computed value along a single crafted path', () => {
    const w = new Float32Array(GENOME_SIZE);
    // hidden[0] = tanh(a*input0 + b); all other hidden units stay 0.
    const a = 0.5;
    const b = 0.1;
    w[0] = a; // input0 -> hidden0
    w[INPUT_SIZE] = b; // hidden0 bias (index 8)
    // output[0] = tanh(c*hidden0 + d); output[1] stays 0.
    const c = 0.8;
    const d = -0.2;
    const outBase = HIDDEN_SIZE * (INPUT_SIZE + 1); // 90
    w[outBase] = c; // hidden0 -> output0
    w[outBase + HIDDEN_SIZE] = d; // output0 bias (index 100)

    const s = scratch();
    s.inputs[0] = 1.0;
    forward(w, 0, s.inputs, s.hidden, s.outputs);

    const expectedHidden0 = Math.tanh(a * 1.0 + b);
    const expectedOut0 = Math.tanh(c * expectedHidden0 + d);
    expect(s.hidden[0]).toBeCloseTo(expectedHidden0, 6);
    expect(s.outputs[0]).toBeCloseTo(expectedOut0, 6);
    expect(s.outputs[1]).toBeCloseTo(0, 6);
  });

  it('honours the weight offset (operates on a slice of a population buffer)', () => {
    const w = new Float32Array(GENOME_SIZE * 2);
    const s = scratch();
    s.inputs.fill(0.3);
    // Genome 0 stays zero; genome 1 gets a bias on output0.
    const outBase = HIDDEN_SIZE * (INPUT_SIZE + 1);
    w[GENOME_SIZE + outBase + HIDDEN_SIZE] = 0.5;

    forward(w, 0, s.inputs, s.hidden, s.outputs);
    expect(s.outputs[0]).toBeCloseTo(0, 6);

    forward(w, GENOME_SIZE, s.inputs, s.hidden, s.outputs);
    expect(s.outputs[0]).toBeCloseTo(Math.tanh(0.5), 6);
  });

  it('disables a hidden neuron when its activation gene is negative', () => {
    const w = new Float32Array(GENOME_SIZE);
    // Craft a single hidden0 -> output0 path.
    w[0] = 0.5; // input0 -> hidden0
    w[INPUT_SIZE] = 0.1; // hidden0 bias
    const outBase = HIDDEN_SIZE * (INPUT_SIZE + 1);
    w[outBase] = 0.8; // hidden0 -> output0
    w[outBase + HIDDEN_SIZE] = -0.2; // output0 bias
    const s = scratch();
    s.inputs[0] = 1.0;

    // Gene >= 0 (zero default): neuron active, output reflects the path.
    forward(w, 0, s.inputs, s.hidden, s.outputs);
    expect(s.hidden[0]!).toBeGreaterThan(0);
    expect(s.outputs[0]).toBeCloseTo(Math.tanh(0.8 * Math.tanh(0.6) - 0.2), 6);

    // Gene < 0: neuron disabled, only the output bias remains.
    w[WEIGHT_GENES] = -1;
    forward(w, 0, s.inputs, s.hidden, s.outputs);
    expect(s.hidden[0]).toBe(0);
    expect(s.outputs[0]).toBeCloseTo(Math.tanh(-0.2), 6);
  });

  it('keeps outputs in [-1, 1]', () => {
    const rng = new Rng(5);
    const w = new Float32Array(GENOME_SIZE);
    randomGenome(w, 0, rng, 5); // large weights to push toward saturation
    const s = scratch();
    for (let t = 0; t < 100; t++) {
      for (let i = 0; i < INPUT_SIZE; i++) s.inputs[i] = rng.nextRange(-1, 1);
      forward(w, 0, s.inputs, s.hidden, s.outputs);
      for (let o = 0; o < OUTPUT_SIZE; o++) {
        expect(s.outputs[o]).toBeGreaterThanOrEqual(-1);
        expect(s.outputs[o]).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('genome', () => {
  it('randomGenome is reproducible from the same rng state', () => {
    const a = new Float32Array(GENOME_SIZE);
    const b = new Float32Array(GENOME_SIZE);
    randomGenome(a, 0, new Rng(1), 1);
    randomGenome(b, 0, new Rng(1), 1);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('mutateGenome with rate 0 is an exact copy', () => {
    const src = new Float32Array(GENOME_SIZE);
    randomGenome(src, 0, new Rng(2), 1);
    const dst = new Float32Array(GENOME_SIZE);
    mutateGenome(src, 0, dst, 0, new Rng(9), 0, 0.3);
    expect(Array.from(dst)).toEqual(Array.from(src));
  });

  it('mutateGenome is reproducible and mutates ~rate of weights', () => {
    const src = new Float32Array(GENOME_SIZE);
    randomGenome(src, 0, new Rng(3), 1);

    const dst1 = new Float32Array(GENOME_SIZE);
    const dst2 = new Float32Array(GENOME_SIZE);
    mutateGenome(src, 0, dst1, 0, new Rng(7), 0.1, 0.3);
    mutateGenome(src, 0, dst2, 0, new Rng(7), 0.1, 0.3);
    expect(Array.from(dst1)).toEqual(Array.from(dst2));

    // Statistical: over many genomes, fraction changed ~ rate.
    let changed = 0;
    let total = 0;
    const rng = new Rng(123);
    for (let g = 0; g < 200; g++) {
      const out = new Float32Array(GENOME_SIZE);
      mutateGenome(src, 0, out, 0, rng, 0.1, 0.3);
      for (let i = 0; i < GENOME_SIZE; i++) {
        total++;
        if (out[i] !== src[i]) changed++;
      }
    }
    expect(changed / total).toBeCloseTo(0.1, 1);
  });
});
