// Grid build + per-tick clears (share life_common.wgsl bindings).

// Zero per-cell counts (only the [0, numCells) region of gridData).
@compute @workgroup_size(256)
fn gridClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= P.d0.z) { return; }
  atomicStore(&gridData[c], 0u);
}

// Reset per-food eat claims for this tick.
@compute @workgroup_size(256)
fn claimClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d1.x) { return; }
  atomicStore(&gridData[claimIdx(j)], 0u);
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

// Exclusive prefix sum of counts -> cellStart, reusing gridData[c] as the
// scatter cursor, and reset the food spawn budget for this tick. Serial in one
// invocation (numCells is modest). freeCount (gridData[numCells]) persists.
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
  atomicStore(&gridData[budgetIdx()], u32(P.p3.w)); // food spawn budget
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
