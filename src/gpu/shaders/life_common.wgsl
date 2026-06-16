// Shared bindings + helpers for the full GPU life cycle (Phase 2.0).
// Two bind groups keep each under the storage-buffer-per-stage limit.

struct Params {
  p0: vec4<f32>, // worldW, worldH, cellSize, maxSpeed
  p1: vec4<f32>, // maxTurn, dt, eatRadius, foodEnergy
  p2: vec4<f32>, // baseCost, moveCost, reproThreshold, initialEnergy
  p3: vec4<f32>, // mutationRate, mutationStd, offspringFraction, _
  d0: vec4<u32>, // cols, rows, numCells, n
  d1: vec4<u32>, // f (food count), frame, _, _
};

// Group 0: creature + world state.
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4<f32>>; // x,y,heading,speed
@group(0) @binding(2) var<storage, read_write> bio: array<vec4<f32>>;   // energy, hue, alive, age
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;     // genomes
@group(0) @binding(4) var<storage, read_write> foodPos: array<vec2<f32>>;

// Group 1: grid + lifecycle scratch. To fit maxStorageBuffersPerShaderStage (8),
// the per-cell counts (reused as the scatter cursor), the lifecycle counters and
// the per-food eat-claims are packed into one atomic buffer `gridData`:
//   [0, numCells)              -> per-cell count, then reused as scatter cursor
//   [numCells]                 -> freeCount (free-slot stack size; persists)
//   [numCells + 1]             -> food spawn budget for this tick
//   [numCells + 2, +f)         -> per-food eat claim (1 = taken this tick)
@group(1) @binding(0) var<storage, read_write> gridData: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> cellStart: array<u32>; // offsets, numCells+1
@group(1) @binding(2) var<storage, read_write> sortedIdx: array<u32>; // food indices by cell
@group(1) @binding(3) var<storage, read_write> freeList: array<u32>;  // free slot indices

fn freeCountIdx() -> u32 { return P.d0.z; }
fn budgetIdx() -> u32 { return P.d0.z + 1u; }
fn claimIdx(j: u32) -> u32 { return P.d0.z + 2u + j; }

const GENOME_SIZE: u32 = 112u;
const NONE: u32 = 0xffffffffu;
const TAU: f32 = 6.2831853;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(a: u32, b: u32) -> f32 {
  return f32(pcg(pcg(a) ^ b)) / 4294967296.0;
}
fn gaussian(a: u32, b: u32) -> f32 {
  let u1 = max(rnd(a, b), 1e-7);
  let u2 = rnd(a ^ 0x9e3779b9u, b);
  return sqrt(-2.0 * log(u1)) * cos(TAU * u2);
}

fn wrapf(x: f32, s: f32) -> f32 {
  return x - floor(x / s) * s;
}
fn wrapHue(h: f32) -> f32 {
  return h - floor(h);
}
fn wrapDelta(d: f32, s: f32) -> f32 {
  let h = s * 0.5;
  if (d > h) { return d - s; }
  if (d < -h) { return d + s; }
  return d;
}
fn cellOf(x: f32, y: f32) -> u32 {
  let cs = P.p0.z;
  let cols = P.d0.x;
  let rows = P.d0.y;
  var cx = u32(floor(x / cs));
  var cy = u32(floor(y / cs));
  if (cx >= cols) { cx = cols - 1u; }
  if (cy >= rows) { cy = rows - 1u; }
  return cy * cols + cx;
}
