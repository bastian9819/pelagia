// Per-creature step: sense nearest food + nearest neighbour via the grids ->
// brain -> move -> eat food (atomic claim) -> prey on a smaller neighbour
// (atomic claim) -> metabolise. Only alive creatures act.

const INPUT_SIZE: u32 = 8u;
const HIDDEN_SIZE: u32 = 10u;
const OUTPUT_SIZE: u32 = 2u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.d0.w) { return; }
  var b = bio[i];
  if (b.z < 0.5) { return; } // dead / free slot

  let s = state[i];
  let size = creatureSize(i); // body size: bigger preys, but slower + costlier
  let W = P.p0.x;
  let H = P.p0.y;
  let cs = P.p0.z;
  let cols = i32(P.d0.x);
  let rows = i32(P.d0.y);

  let bcx = i32(floor(s.x / cs));
  let bcy = i32(floor(s.y / cs));

  // Nearest food across the 3x3 cell block (cellSize == perception radius).
  var bestD2 = 1.0e30;
  var bestIdx = NONE;
  // Nearest OTHER creature across the same 3x3 block (creature grid).
  var nbrD2 = 1.0e30;
  var nbrIdx = NONE;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    var cy = (bcy + dy) % rows;
    if (cy < 0) { cy = cy + rows; }
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      var cx = (bcx + dx) % cols;
      if (cx < 0) { cx = cx + cols; }
      let cell = u32(cy) * P.d0.x + u32(cx);
      // food in this cell
      let fstart = cellStart[cell];
      let fend = cellStart[cell + 1u];
      for (var k = fstart; k < fend; k = k + 1u) {
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
      // creatures in this cell
      let cstart = cellStart[creatureStartBase() + cell];
      let cend = cellStart[creatureStartBase() + cell + 1u];
      for (var k = cstart; k < cend; k = k + 1u) {
        let nj = sortedIdx[creatureSortBase() + k];
        if (nj == i) { continue; } // skip self
        let np = state[nj];
        let ddx = wrapDelta(np.x - s.x, W);
        let ddy = wrapDelta(np.y - s.y, H);
        let d2 = ddx * ddx + ddy * ddy;
        if (d2 < nbrD2) {
          nbrD2 = d2;
          nbrIdx = nj;
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
  if (nbrIdx != NONE && nbrD2 <= cs * cs) {
    let np = state[nbrIdx];
    let ndx = wrapDelta(np.x - s.x, W);
    let ndy = wrapDelta(np.y - s.y, H);
    let nrel = atan2(ndy, ndx) - s.z;
    inp[3] = cos(nrel);
    inp[4] = sin(nrel);
    inp[5] = max(0.0, 1.0 - sqrt(nbrD2) / cs);
  } else {
    inp[3] = 0.0;
    inp[4] = 0.0;
    inp[5] = 0.0;
  }
  inp[6] = b.x / P.p2.z; // energy / reproThreshold
  inp[7] = s.w / P.p0.w; // speed / maxSpeed

  // Brain forward pass. Disabled neurons (activation gene < 0) contribute nothing.
  var p = i * GENOME_SIZE;
  let actBase = i * GENOME_SIZE + WEIGHT_GENES;
  var hidden: array<f32, HIDDEN_SIZE>;
  for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
    var sum = 0.0;
    for (var k = 0u; k < INPUT_SIZE; k = k + 1u) {
      sum = sum + weights[p] * inp[k];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    hidden[h] = select(0.0, tanh(sum), weights[actBase + h] >= 0.0);
  }
  var out: array<f32, OUTPUT_SIZE>;
  for (var o = 0u; o < OUTPUT_SIZE; o = o + 1u) {
    var sum = 0.0;
    for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
      sum = sum + weights[p] * hidden[h];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    out[o] = tanh(sum);
  }

  // Move. Bigger bodies have a lower top speed (prey can outrun predators).
  let heading = s.z + out[0] * P.p1.x;
  let maxSp = P.p0.w / sqrt(size);
  let speed = (out[1] + 1.0) * 0.5 * maxSp;
  let nx = wrapf(s.x + cos(heading) * speed * P.p1.y, W);
  let ny = wrapf(s.y + sin(heading) * speed * P.p1.y, H);
  state[i] = vec4<f32>(nx, ny, heading, speed);

  var energy = b.x;
  let eatR = P.p1.z;

  // Eat: claim the nearest food if it is within eat radius (one winner).
  if (bestIdx != NONE) {
    let fp = foodPos[bestIdx];
    let ddx = wrapDelta(fp.x - nx, W);
    let ddy = wrapDelta(fp.y - ny, H);
    if (ddx * ddx + ddy * ddy <= eatR * eatR) {
      let prev = atomicExchange(&gridData[claimIdx(bestIdx)], 1u);
      if (prev == 0u) {
        energy = energy + P.p1.w; // foodEnergy
        foodPos[bestIdx] = vec2<f32>(-1.0, -1.0); // mark eaten (rate-limited respawn)
      }
    }
  }

  // Prey on the nearest neighbour if we're enough BIGGER (by body size) and in
  // contact. The size margin makes mutual predation impossible (can't both be
  // 1.25x bigger), so a single atomic claim per prey is enough; the predator
  // credits ITS OWN energy here and the prey is freed in the death pass
  // (race-free). Reach scales with size, so big slow hunters can still strike.
  let gain = P.ext.x;
  if (gain > 0.0 && nbrIdx != NONE) {
    let preySize = creatureSize(nbrIdx);
    let reach = eatR * size;
    let np = state[nbrIdx];
    let pdx = wrapDelta(np.x - nx, W);
    let pdy = wrapDelta(np.y - ny, H);
    if (pdx * pdx + pdy * pdy <= reach * reach && size > preySize * P.ext.y) {
      let claim = atomicCompareExchangeWeak(&gridData[eatenIdx(nbrIdx)], 0u, i + 1u);
      if (claim.exchanged) {
        energy = energy + gain * max(0.0, bio[nbrIdx].x);
        atomicAdd(&gridData[predCountIdx()], 1u);
      }
    }
  }

  // Metabolise. Bigger bodies cost more just to stay alive (size's downside).
  energy = energy - (P.p2.x * size + P.p2.y * speed);
  b.x = energy;
  // bio.w is the stable lineage id (set at birth, inherited) — never modified here.
  bio[i] = b;
}
