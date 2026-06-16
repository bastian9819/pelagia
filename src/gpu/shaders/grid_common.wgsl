// Shared bindings + helpers for the GPU spatial grid (counting sort) and the
// sensing/brain/move step. Concatenated in front of each pass's entry point.
// One explicit bind group is shared across all grid passes.

struct GridParams {
  worldSize: vec4<f32>, // w, h, cellSize, maxSpeed
  motion: vec4<f32>,    // maxTurn, dt, _, _
  dims: vec4<u32>,      // cols, rows, numCells, n
  counts: vec4<u32>,    // f (food count), frame, _, _
};

@group(0) @binding(0) var<uniform> GP: GridParams;
@group(0) @binding(1) var<storage, read_write> state: array<vec4<f32>>;   // x,y,heading,speed
@group(0) @binding(2) var<storage, read> weights: array<f32>;             // genomes
@group(0) @binding(3) var<storage, read> foodPos: array<vec2<f32>>;       // food positions
@group(0) @binding(4) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellStart: array<u32>;     // length numCells+1
@group(0) @binding(6) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sortedIdx: array<u32>;     // food indices by cell
@group(0) @binding(8) var<storage, read_write> nearestOut: array<i32>;    // debug/validation

fn cellOf(x: f32, y: f32) -> u32 {
  let cs = GP.worldSize.z;
  let cols = GP.dims.x;
  let rows = GP.dims.y;
  var cx = u32(floor(x / cs));
  var cy = u32(floor(y / cs));
  if (cx >= cols) { cx = cols - 1u; }
  if (cy >= rows) { cy = rows - 1u; }
  return cy * cols + cx;
}

fn wrapDelta(d: f32, s: f32) -> f32 {
  let h = s * 0.5;
  if (d > h) { return d - s; }
  if (d < -h) { return d + s; }
  return d;
}
