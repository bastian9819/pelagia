// Death, reproduction and food respawn (share life_common.wgsl bindings).
// Run as separate dispatches in this order within one compute pass; WebGPU
// synchronises storage writes between dispatches, so repro sees death's frees.

// Death: alive creatures out of energy — or eaten by a predator this tick — free
// their slot onto the stack. (Predation claims are made atomically in the sim
// pass; the predator already credited its own energy there.)
@compute @workgroup_size(256)
fn death(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.d0.w) { return; }
  var b = bio[i];
  if (b.z < 0.5) { return; } // already free
  let eaten = atomicLoad(&gridData[eatenIdx(i)]) != 0u;
  if (b.x <= 0.0 || eaten) {
    b.z = 0.0;
    b.x = 0.0;
    bio[i] = b;
    let slot = atomicAdd(&gridData[freeCountIdx()], 1u);
    freeList[slot] = i;
  }
}

// Reproduction: alive creatures above the energy threshold pop a free slot
// (lock-free CAS) and write a mutated offspring into it.
@compute @workgroup_size(256)
fn repro(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.d0.w) { return; }
  var b = bio[i];
  if (b.z < 0.5) { return; }
  // Bigger bodies need proportionally more energy to reproduce, so small bodies
  // out-breed big ones — a fecundity counterweight to predation's size advantage,
  // which keeps size from running away and lets small/large niches coexist.
  if (b.x < P.p2.z * creatureSize(i)) { return; } // reproThreshold * size

  var slot = NONE;
  loop {
    let cur = atomicLoad(&gridData[freeCountIdx()]);
    if (cur == 0u) { break; }
    let r = atomicCompareExchangeWeak(&gridData[freeCountIdx()], cur, cur - 1u);
    if (r.exchanged) {
      slot = freeList[cur - 1u];
      break;
    }
  }
  if (slot == NONE) { return; } // no room this tick

  let frame = P.d1.y;
  let half = b.x * P.p3.z; // offspringFraction
  b.x = b.x - half;
  bio[i] = b;

  // Speciation: with probability ext2.z the offspring FOUNDS a new lineage — a new
  // stable id, a fresh colour (visible split) and a pointer to its parent lineage,
  // so descent forms a branching tree. Otherwise it inherits hue + lineage as-is.
  var childHue = b.y;
  var childLin = b.w;
  if (P.ext2.z > 0.0 && rnd(i + 7u, frame) < P.ext2.z) {
    let cur = atomicAdd(&gridData[speciesCountIdx()], 1u);
    if (cur < P.d0.w) { // cap new lineages at n
      let nid = P.d0.w + cur;
      childLin = f32(nid);
      childHue = f32(pcg(nid)) / 4294967296.0; // matches floatFromU32(pcgHash(nid)) on CPU
      atomicStore(&gridData[parentIdx(cur)], u32(b.w));
    }
  }
  bio[slot] = vec4<f32>(half, childHue, 1.0, childLin);

  let s = state[i];
  let jx = (rnd(i + 11u, frame) - 0.5) * 4.0;
  let jy = (rnd(i + 29u, frame) - 0.5) * 4.0;
  let hd = rnd(i + 53u, frame) * TAU;
  state[slot] = vec4<f32>(wrapf(s.x + jx, P.p0.x), wrapf(s.y + jy, P.p0.y), hd, 0.0);

  let src = i * GENOME_SIZE;
  let dst = slot * GENOME_SIZE;
  let rate = P.p3.x;
  let mutStd = P.p3.y;
  for (var k = 0u; k < GENOME_SIZE; k = k + 1u) {
    var w = weights[src + k];
    if (rnd(i * 131u + k, frame) < rate) {
      w = w + gaussian(i * 977u + k, frame) * mutStd;
    }
    weights[dst + k] = w;
  }
}

// Rate-limited food spawn: dead food slots compete for a per-tick budget
// (gridData[budgetIdx], set in scan). Caps energy influx -> carrying capacity
// below N -> turnover -> selection (the CPU ecology, on the GPU).
@compute @workgroup_size(256)
fn foodRespawn(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d1.x) { return; }
  if (foodPos[j].x >= 0.0) { return; } // already alive

  loop {
    let cur = atomicLoad(&gridData[budgetIdx()]);
    if (cur == 0u) { return; }
    let r = atomicCompareExchangeWeak(&gridData[budgetIdx()], cur, cur - 1u);
    if (r.exchanged) { break; }
  }
  let frame = P.d1.y;
  // Two food types by index. Big food (rare, low indices) respawns in a FEW tight
  // drifting blooms -> rich hotspots worth competing over. Plankton (the rest)
  // clusters per the patchiness slider (P.ext.z): uniform (0) -> tight (1).
  let big = j < (P.d1.x / 16u);
  var K = 5u;
  var spread = mix(P.p0.x * 0.5, P.p0.x * 0.04, clamp(P.ext.z, 0.0, 1.0));
  var drift = 0.0008;
  if (big) {
    K = 3u;
    spread = P.p0.x * 0.05; // big-food blooms are always tight
    drift = 0.0006;
  }
  let bk = pcg(j) % K;
  let ang = f32(bk) * 2.3999632 + f32(frame) * drift;
  let cx = P.p0.x * (0.5 + 0.32 * cos(ang));
  let cy = P.p0.y * (0.5 + 0.32 * sin(ang * 1.3 + f32(bk)));
  let fx = cx + (rnd(j + 17u, frame) - 0.5) * 2.0 * spread;
  let fy = cy + (rnd(j + 83u, frame) - 0.5) * 2.0 * spread;
  foodPos[j] = vec2<f32>(wrapf(fx, P.p0.x), wrapf(fy, P.p0.y));
}
