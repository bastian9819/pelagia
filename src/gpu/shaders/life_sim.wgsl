// Per-creature step: sense nearest food + nearest neighbour via the grids ->
// brain -> move -> eat food (atomic claim) -> prey on a smaller neighbour
// (atomic claim) -> metabolise. Only alive creatures act.

const INPUT_SIZE: u32 = 11u;
const HIDDEN_SIZE: u32 = 10u;
const OUTPUT_SIZE: u32 = 3u;

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

  // Nearest food of EACH type across the 3x3 cell block (cellSize == perception
  // radius). Big food = low indices [0, f/16); plankton = the rest. Two separate
  // channels let a brain steer toward its preferred resource even when the other
  // type is closer — that is what makes a real sensory specialist (vs the spatial
  // niche of D-023). Eating still targets the single nearest pellet (min of both).
  let fBig = u32(f32(P.d1.x) * P.ext3.x); // big-food slot count (ext3.x fraction; 0 = none)
  var pD2 = 1.0e30; // nearest plankton
  var pIdx = NONE;
  var gD2 = 1.0e30; // nearest big food
  var gIdx = NONE;
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
        if (fj < fBig) {
          if (d2 < gD2) { gD2 = d2; gIdx = fj; }
        } else {
          if (d2 < pD2) { pD2 = d2; pIdx = fj; }
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
  // Plankton channel.
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
  inp[9] = b.x / P.p2.z; // energy / reproThreshold
  inp[10] = s.w / P.p0.w; // speed / maxSpeed

  // For eating, target the single nearest pellet of either type.
  var bestIdx = NONE;
  var bestD2 = 1.0e30;
  if (pIdx != NONE && pD2 < bestD2) { bestD2 = pD2; bestIdx = pIdx; }
  if (gIdx != NONE && gD2 < bestD2) { bestD2 = gD2; bestIdx = gIdx; }

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
  // Elongation is a real morphology trade-off: streamlined eels (elong>1) swim
  // faster but turn worse; round blobs (elong<1) are nimble but slower.
  let elong = creatureElong(i);
  let et = (elong - 0.5) / 1.5; // 0 = round, 1 = eel
  let heading = s.z + out[0] * P.p1.x * mix(1.2, 0.8, et);
  let maxSp = (P.p0.w / sqrt(size)) * mix(0.85, 1.2, et);
  let speed = (out[1] + 1.0) * 0.5 * maxSp;
  // Ocean current advects every creature (ext3.w strength; 0 = still water), so
  // they must cope with drift — gyres, downstream pile-ups, harder upstream foraging.
  let cur = currentAt(s.x, s.y, f32(P.d1.y)) * P.ext3.w;
  let nx = wrapf(s.x + cos(heading) * speed * P.p1.y + cur.x, W);
  let ny = wrapf(s.y + sin(heading) * speed * P.p1.y + cur.y, H);
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
        // Big food (low indices) is worth ext.w times a plankton pellet.
        let big = bestIdx < fBig;
        energy = energy + P.p1.w * select(1.0, P.ext.w, big);
        foodPos[bestIdx] = vec2<f32>(-1.0, -1.0); // mark eaten (rate-limited respawn)
      }
    }
  }

  // Prey on the nearest neighbour, but ONLY if the brain CHOSE to attack this
  // tick (out[2] > 0) and we're enough BIGGER (body size). Predation is now an
  // evolved DECISION, not an automatic size rule — hunter lineages learn to fire
  // their attack output near smaller neighbours; foragers leave it off. The size
  // margin still rules out mutual predation, so a single atomic claim per prey is
  // enough; the predator credits ITS OWN energy here and the prey is freed in the
  // death pass (race-free). Reach scales with size, so big slow hunters can strike.
  let attacking = out[2] > 0.0;
  let gain = P.ext.x;
  if (gain > 0.0 && attacking && nbrIdx != NONE) {
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

  // Metabolise. Bigger bodies cost more just to stay alive (size's downside);
  // turning costs energy too (ext2.w; 0 = free) so agile steering isn't free,
  // bioluminescence can cost energy (ext3.z; 0 = free), and attacking costs energy
  // each tick the attack output is on (ext4.x; 0 = free) so indiscriminate
  // aggression is selected against — the lunge has a price whether or not it lands.
  let glowCost = P.ext3.z * max(0.0, creatureGlow(i) - 1.0);
  let attackCost = select(0.0, P.ext4.x, attacking);
  energy = energy - (P.p2.x * size + P.p2.y * speed + P.ext2.w * abs(out[0]) + glowCost + attackCost);
  b.x = energy;
  // bio.w is the stable lineage id (set at birth, inherited) — never modified here.
  bio[i] = b;
}
