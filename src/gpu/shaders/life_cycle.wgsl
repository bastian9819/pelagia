// Death, reproduction and food respawn (share life_common.wgsl bindings).
// Run as separate dispatches in this order within one compute pass; WebGPU
// synchronises storage writes between dispatches, so repro sees death's frees.

// Death: alive creatures out of energy free their slot onto the stack.
@compute @workgroup_size(256)
fn death(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.d0.w) { return; }
  var b = bio[i];
  if (b.z < 0.5) { return; } // already free
  if (b.x <= 0.0) {
    b.z = 0.0;
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
  if (b.x < P.p2.z) { return; } // reproThreshold

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

  bio[slot] = vec4<f32>(half, wrapHue(b.y + gaussian(i, frame) * 0.015), 1.0, 0.0);

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
  foodPos[j] = vec2<f32>(rnd(j + 17u, frame) * P.p0.x, rnd(j + 83u, frame) * P.p0.y);
}
