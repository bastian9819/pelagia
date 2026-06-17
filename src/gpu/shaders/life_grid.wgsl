// Grid build + per-tick clears (share life_common.wgsl bindings).
// Phase 6: a second grid over CREATURES (for neighbour perception + predation)
// is built into the same buffers as the food grid.

// Zero per-cell counts for BOTH grids ([0, 2*numCells) region of gridData).
@compute @workgroup_size(256)
fn gridClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= P.d0.z * 2u) { return; }
  atomicStore(&gridData[c], 0u);
}

// Reset per-food eat claims and per-creature predation claims for this tick.
// (food slots f == creature slots n here, so one dispatch over f clears both.)
@compute @workgroup_size(256)
fn claimClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d1.x) { return; }
  atomicStore(&gridData[claimIdx(j)], 0u);
  atomicStore(&gridData[eatenIdx(j)], 0u);
}

// Count alive food pellets per cell.
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d1.x) { return; }
  let p = foodPos[j];
  if (p.x < 0.0) { return; } // dead food is not in the grid
  atomicAdd(&gridData[cellOf(p.x, p.y)], 1u);
}

// Count alive creatures per cell (second grid).
@compute @workgroup_size(256)
fn countCreatures(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.d0.w) { return; }
  if (bio[i].z < 0.5) { return; } // dead slot
  let s = state[i];
  atomicAdd(&gridData[creatureCountIdx(cellOf(s.x, s.y))], 1u);
}

// Exclusive prefix sums of BOTH count arrays -> cellStart (food region then
// creature region), reusing the count slots as scatter cursors. Also resets the
// food spawn budget and predation counter. Serial in one invocation.
@compute @workgroup_size(1)
fn scan() {
  let nc = P.d0.z;
  var acc = 0u;
  for (var c = 0u; c < nc; c = c + 1u) {
    let cnt = atomicLoad(&gridData[c]);
    cellStart[c] = acc;
    atomicStore(&gridData[c], acc); // reuse cell slot as cursor
    acc = acc + cnt;
  }
  cellStart[nc] = acc;

  let base = creatureStartBase();
  var cacc = 0u;
  for (var c = 0u; c < nc; c = c + 1u) {
    let cnt = atomicLoad(&gridData[creatureCountIdx(c)]);
    cellStart[base + c] = cacc;
    atomicStore(&gridData[creatureCountIdx(c)], cacc);
    cacc = cacc + cnt;
  }
  cellStart[base + nc] = cacc;

  // Day/night cycle: food influx swings with a slow sine so the ocean has rich
  // and lean times (boom/bust). ext2.x = strength (0 disables), ext2.y = period.
  let phase = sin(6.2831853 * f32(P.d1.y) / max(1.0, P.ext2.y));
  let dayFactor = max(0.1, 1.0 + P.ext2.x * phase);
  atomicStore(&gridData[budgetIdx()], u32(P.p3.w * dayFactor)); // food spawn budget
  atomicStore(&gridData[predCountIdx()], 0u); // predation kills this tick
}

// Scatter alive food indices into their cell buckets.
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d1.x) { return; }
  let p = foodPos[j];
  if (p.x < 0.0) { return; }
  let dst = atomicAdd(&gridData[cellOf(p.x, p.y)], 1u);
  sortedIdx[dst] = j;
}

// Scatter alive creature indices into their cell buckets (creature region).
@compute @workgroup_size(256)
fn scatterCreatures(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.d0.w) { return; }
  if (bio[i].z < 0.5) { return; }
  let s = state[i];
  let dst = atomicAdd(&gridData[creatureCountIdx(cellOf(s.x, s.y))], 1u);
  sortedIdx[creatureSortBase() + dst] = i;
}
