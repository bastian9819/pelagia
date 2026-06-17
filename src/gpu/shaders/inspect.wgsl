// Brain inspector: for one selected creature, recompute its sensors (nearest
// food via the grid) and full forward pass, and write inputs/hidden/outputs +
// state into a tiny buffer for read-back. Standalone bind group (<= 8 storage
// buffers). The selected slot index is passed in P.d1.z.

struct Params {
  p0: vec4<f32>, // worldW, worldH, cellSize, maxSpeed
  p1: vec4<f32>, // maxTurn, dt, eatRadius, foodEnergy
  p2: vec4<f32>, // baseCost, moveCost, reproThreshold, initialEnergy
  p3: vec4<f32>,
  d0: vec4<u32>, // cols, rows, numCells, n
  d1: vec4<u32>, // f, frame, selectedIndex, _
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bio: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> foodPos: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> cellStart: array<u32>;
@group(0) @binding(6) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

const INPUT_SIZE: u32 = 8u;
const HIDDEN_SIZE: u32 = 10u;
const OUTPUT_SIZE: u32 = 2u;
const GENOME_SIZE: u32 = 112u;
const NONE: u32 = 0xffffffffu;

fn wrapDelta(d: f32, s: f32) -> f32 {
  let h = s * 0.5;
  if (d > h) { return d - s; }
  if (d < -h) { return d + s; }
  return d;
}

@compute @workgroup_size(1)
fn main() {
  let i = P.d1.z;
  if (i >= P.d0.w) { return; }
  let s = state[i];
  let b = bio[i];
  let W = P.p0.x;
  let H = P.p0.y;
  let cs = P.p0.z;
  let cols = i32(P.d0.x);
  let rows = i32(P.d0.y);

  let bcx = i32(floor(s.x / cs));
  let bcy = i32(floor(s.y / cs));
  var bestD2 = 1.0e30;
  var bestIdx = NONE;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    var cy = (bcy + dy) % rows;
    if (cy < 0) { cy = cy + rows; }
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      var cx = (bcx + dx) % cols;
      if (cx < 0) { cx = cx + cols; }
      let cell = u32(cy) * P.d0.x + u32(cx);
      let start = cellStart[cell];
      let end = cellStart[cell + 1u];
      for (var k = start; k < end; k = k + 1u) {
        let fj = sortedIdx[k];
        let fp = foodPos[fj];
        let ddx = wrapDelta(fp.x - s.x, W);
        let ddy = wrapDelta(fp.y - s.y, H);
        let d2 = ddx * ddx + ddy * ddy;
        if (d2 < bestD2) {
          bestD2 = d2;
          bestIdx = fj;
        }
      }
    }
  }

  var inp: array<f32, INPUT_SIZE>;
  if (bestIdx != NONE && bestD2 <= cs * cs) {
    let fp = foodPos[bestIdx];
    let ddx = wrapDelta(fp.x - s.x, W);
    let ddy = wrapDelta(fp.y - s.y, H);
    let rel = atan2(ddy, ddx) - s.z;
    inp[0] = cos(rel);
    inp[1] = sin(rel);
    inp[2] = max(0.0, 1.0 - sqrt(bestD2) / cs);
  } else {
    inp[0] = 0.0;
    inp[1] = 0.0;
    inp[2] = 0.0;
  }
  inp[3] = 0.0;
  inp[4] = 0.0;
  inp[5] = 0.0;
  inp[6] = b.x / P.p2.z;
  inp[7] = s.w / P.p0.w;

  var p = i * GENOME_SIZE;
  var hidden: array<f32, HIDDEN_SIZE>;
  for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
    var sum = 0.0;
    for (var k = 0u; k < INPUT_SIZE; k = k + 1u) {
      sum = sum + weights[p] * inp[k];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    hidden[h] = tanh(sum);
  }
  var outv: array<f32, OUTPUT_SIZE>;
  for (var o = 0u; o < OUTPUT_SIZE; o = o + 1u) {
    var sum = 0.0;
    for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
      sum = sum + weights[p] * hidden[h];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    outv[o] = tanh(sum);
  }

  // Write inputs(8) | hidden(10) | outputs(2) | x,y,heading,speed,energy,hue,age,alive
  for (var k = 0u; k < INPUT_SIZE; k = k + 1u) { out[k] = inp[k]; }
  for (var k = 0u; k < HIDDEN_SIZE; k = k + 1u) { out[8u + k] = hidden[k]; }
  out[18] = outv[0];
  out[19] = outv[1];
  out[20] = s.x;
  out[21] = s.y;
  out[22] = s.z;
  out[23] = s.w;
  out[24] = b.x;
  out[25] = b.y;
  out[26] = b.w;
  out[27] = b.z;
}
