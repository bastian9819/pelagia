// Phase 1.0 movement compute shader: a simple per-creature wander.
// No neighbour reads, so each thread only touches its own element (no
// ping-pong needed yet). Establishes the compute+buffer ceiling at scale.

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

// PCG hash (matches the CPU reference rng so behaviour is comparable).
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
  let turn = (rnd(i, P.frame) - 0.5) * 2.0 * P.maxTurn;
  let heading = s.z + turn;
  let speed = P.maxSpeed * (0.5 + 0.5 * rnd(i + 1234567u, P.frame));

  var x = s.x + cos(heading) * speed * P.dt;
  var y = s.y + sin(heading) * speed * P.dt;
  x = x - floor(x / P.width) * P.width;
  y = y - floor(y / P.height) * P.height;

  state[i] = vec4<f32>(x, y, heading, speed);
}
