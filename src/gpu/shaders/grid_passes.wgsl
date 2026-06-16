// Counting-sort grid build passes (share grid_common.wgsl bindings).

// 1) Zero the per-cell counts.
@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= GP.dims.z) { return; }
  atomicStore(&cellCount[c], 0u);
}

// 2) Count food pellets per cell.
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= GP.counts.x) { return; }
  let p = foodPos[j];
  atomicAdd(&cellCount[cellOf(p.x, p.y)], 1u);
}

// 3) Exclusive prefix sum of cellCount -> cellStart, and seed the scatter cursor.
// Serial in a single invocation: numCells is modest (~10^4) so this is a tiny
// fraction of the frame. The production renderer will use a parallel scan.
@compute @workgroup_size(1)
fn scan() {
  let nc = GP.dims.z;
  var acc = 0u;
  for (var c = 0u; c < nc; c = c + 1u) {
    let cnt = atomicLoad(&cellCount[c]);
    cellStart[c] = acc;
    atomicStore(&cursor[c], acc);
    acc = acc + cnt;
  }
  cellStart[nc] = acc;
}

// 4) Scatter food indices into their cell buckets.
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= GP.counts.x) { return; }
  let p = foodPos[j];
  let c = cellOf(p.x, p.y);
  let dst = atomicAdd(&cursor[c], 1u);
  sortedIdx[dst] = j;
}
