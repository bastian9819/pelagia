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
  ext: vec4<f32>,
  ext2: vec4<f32>,
  ext3: vec4<f32>, // bigFoodFraction in .x (so the viewer senses food types like the sim)
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bio: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> foodPos: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> cellStart: array<u32>;
@group(0) @binding(6) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

const INPUT_SIZE: u32 = 13u;
const HIDDEN_SIZE: u32 = 10u;
const OUTPUT_SIZE: u32 = 3u;
const WEIGHT_GENES: u32 = 173u;
const SIZE_GENE: u32 = 183u;
const ELONG_GENE: u32 = 184u;
const GLOW_GENE: u32 = 186u;
const THERMAL_GENE: u32 = 187u;
const GENOME_SIZE: u32 = 188u;
const NONE: u32 = 0xffffffffu;

fn wrapDelta(d: f32, s: f32) -> f32 {
  let h = s * 0.5;
  if (d > h) { return d - s; }
  if (d < -h) { return d + s; }
  return d;
}

// Mirror of life_common's tempAt so the viewer shows the same temperature sensor.
fn tempAt(x: f32, y: f32, frame: f32) -> f32 {
  let TAU = 6.2831853;
  let drift = frame * 0.0004;
  return clamp(0.6 * cos((y / P.p0.y) * TAU + drift) + 0.4 * sin((x / P.p0.x) * TAU), -1.0, 1.0);
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
  let nc = P.d0.z;
  let f = P.d1.x;
  let fBig = u32(f32(f) * P.ext3.x);
  var pD2 = 1.0e30; // nearest plankton
  var pIdx = NONE;
  var gD2 = 1.0e30; // nearest big food
  var gIdx = NONE;
  var nbrD2 = 1.0e30;
  var nbrIdx = NONE;
  var nbrCount = 0.0;
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
        if (fj < fBig) {
          if (d2 < gD2) { gD2 = d2; gIdx = fj; }
        } else {
          if (d2 < pD2) { pD2 = d2; pIdx = fj; }
        }
      }
      // creature region: cellStart[nc+1+cell], sortedIdx[f + k]
      let cstart = cellStart[nc + 1u + cell];
      let cend = cellStart[nc + 1u + cell + 1u];
      for (var k = cstart; k < cend; k = k + 1u) {
        let nj = sortedIdx[f + k];
        if (nj == i) { continue; }
        let np = state[nj];
        let ddx = wrapDelta(np.x - s.x, W);
        let ddy = wrapDelta(np.y - s.y, H);
        let d2 = ddx * ddx + ddy * ddy;
        if (d2 < nbrD2) {
          nbrD2 = d2;
          nbrIdx = nj;
        }
        if (d2 <= cs * cs) { nbrCount = nbrCount + 1.0; }
      }
    }
  }

  var inp: array<f32, INPUT_SIZE>;
  // Plankton channel (mirrors life_sim so the viewer shows the real sensors).
  if (pIdx != NONE && pD2 <= cs * cs) {
    let fp = foodPos[pIdx];
    let rel = atan2(wrapDelta(fp.y - s.y, H), wrapDelta(fp.x - s.x, W)) - s.z;
    inp[0] = cos(rel);
    inp[1] = sin(rel);
    inp[2] = max(0.0, 1.0 - sqrt(pD2) / cs);
  } else {
    inp[0] = 0.0;
    inp[1] = 0.0;
    inp[2] = 0.0;
  }
  // Big-food channel.
  if (gIdx != NONE && gD2 <= cs * cs) {
    let fp = foodPos[gIdx];
    let rel = atan2(wrapDelta(fp.y - s.y, H), wrapDelta(fp.x - s.x, W)) - s.z;
    inp[3] = cos(rel);
    inp[4] = sin(rel);
    inp[5] = max(0.0, 1.0 - sqrt(gD2) / cs);
  } else {
    inp[3] = 0.0;
    inp[4] = 0.0;
    inp[5] = 0.0;
  }
  if (nbrIdx != NONE && nbrD2 <= cs * cs) {
    let np = state[nbrIdx];
    let ndx = wrapDelta(np.x - s.x, W);
    let ndy = wrapDelta(np.y - s.y, H);
    let nrel = atan2(ndy, ndx) - s.z;
    inp[6] = cos(nrel);
    inp[7] = sin(nrel);
    inp[8] = max(0.0, 1.0 - sqrt(nbrD2) / cs);
  } else {
    inp[6] = 0.0;
    inp[7] = 0.0;
    inp[8] = 0.0;
  }
  inp[9] = b.x / P.p2.z;
  inp[10] = s.w / P.p0.w;
  inp[11] = tempAt(s.x, s.y, f32(P.d1.y));
  inp[12] = min(1.0, nbrCount / 10.0);

  var p = i * GENOME_SIZE;
  let actBase = i * GENOME_SIZE + WEIGHT_GENES;
  var hidden: array<f32, HIDDEN_SIZE>;
  var activeCount = 0.0;
  for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
    var sum = 0.0;
    for (var k = 0u; k < INPUT_SIZE; k = k + 1u) {
      sum = sum + weights[p] * inp[k];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    let on = weights[actBase + h] >= 0.0;
    hidden[h] = select(0.0, tanh(sum), on);
    if (on) { activeCount = activeCount + 1.0; }
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

  // Write inputs(11) | hidden(10) | outputs(3) | x,y,heading,speed,energy,hue,lineage,alive
  for (var k = 0u; k < INPUT_SIZE; k = k + 1u) { out[k] = inp[k]; }
  for (var k = 0u; k < HIDDEN_SIZE; k = k + 1u) { out[13u + k] = hidden[k]; }
  out[23] = outv[0];
  out[24] = outv[1];
  out[25] = outv[2]; // attack intent
  out[26] = s.x;
  out[27] = s.y;
  out[28] = s.z;
  out[29] = s.w;
  out[30] = b.x;
  out[31] = b.y;
  out[32] = b.w;
  out[33] = b.z;
  out[34] = activeCount; // Phase 6: how many hidden neurons are switched on
  out[35] = clamp(1.0 + 0.5 * weights[i * GENOME_SIZE + SIZE_GENE], 0.6, 2.2); // body size
  out[36] = clamp(1.0 + 0.6 * weights[i * GENOME_SIZE + ELONG_GENE], 0.5, 2.0); // elongation
  out[37] = clamp(1.0 + 0.6 * weights[i * GENOME_SIZE + GLOW_GENE], 0.6, 2.0); // glow
  out[38] = clamp(weights[i * GENOME_SIZE + THERMAL_GENE], -1.0, 1.0); // thermal preference
}
