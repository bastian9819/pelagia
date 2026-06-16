// Phase 1.1: per-creature neural brain forward pass on the GPU.
// Same fixed MLP as the CPU reference: 8 -> 10 (tanh) -> 2 (tanh), weights read
// from a flat genome buffer. Inputs here are synthesised from self-state (real
// sensing arrives with the spatial grid in 1.2) so this measures the true cost
// of evaluating N brains per tick. GENOME_SIZE must match src/sim/brain.ts.

const INPUT_SIZE: u32 = 8u;
const HIDDEN_SIZE: u32 = 10u;
const OUTPUT_SIZE: u32 = 2u;
const GENOME_SIZE: u32 = 112u;

struct Params {
  width: f32,
  height: f32,
  dt: f32,
  maxSpeed: f32,
  maxTurn: f32,
  frame: u32,
  n: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4<f32>>; // x, y, heading, speed
@group(0) @binding(2) var<storage, read> weights: array<f32>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(a: u32, b: u32) -> f32 {
  return f32(pcg(pcg(a) ^ b)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  let s = state[i];

  // Synthetic egocentric-ish inputs (placeholder until real sensing in 1.2).
  var inp: array<f32, INPUT_SIZE>;
  inp[0] = cos(s.z);
  inp[1] = sin(s.z);
  inp[2] = s.x / P.width * 2.0 - 1.0;
  inp[3] = s.y / P.height * 2.0 - 1.0;
  inp[4] = s.w / P.maxSpeed;
  inp[5] = rnd(i, P.frame) * 2.0 - 1.0;
  inp[6] = rnd(i + 7u, P.frame) * 2.0 - 1.0;
  inp[7] = 1.0;

  var p = i * GENOME_SIZE;
  var hidden: array<f32, HIDDEN_SIZE>;
  for (var h = 0u; h < HIDDEN_SIZE; h++) {
    var sum = 0.0;
    for (var k = 0u; k < INPUT_SIZE; k++) {
      sum += weights[p] * inp[k];
      p++;
    }
    sum += weights[p];
    p++;
    hidden[h] = tanh(sum);
  }
  var out: array<f32, OUTPUT_SIZE>;
  for (var o = 0u; o < OUTPUT_SIZE; o++) {
    var sum = 0.0;
    for (var h = 0u; h < HIDDEN_SIZE; h++) {
      sum += weights[p] * hidden[h];
      p++;
    }
    sum += weights[p];
    p++;
    out[o] = tanh(sum);
  }

  let heading = s.z + out[0] * P.maxTurn;
  let speed = (out[1] + 1.0) * 0.5 * P.maxSpeed;
  var x = s.x + cos(heading) * speed * P.dt;
  var y = s.y + sin(heading) * speed * P.dt;
  x = x - floor(x / P.width) * P.width;
  y = y - floor(y / P.height) * P.height;
  state[i] = vec4<f32>(x, y, heading, speed);
}
