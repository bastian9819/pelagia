// Seed brush ("inseminate"): introduce new creatures by POPPING the free-list,
// exactly like reproduction — so a seeded creature consumes a genuinely free slot
// and the free-list invariant holds (no double-free, no overwriting live creatures,
// no ephemeral seeds). Replaces the old CPU-side write-by-index, which reactivated
// free slots WITHOUT removing them from the free-list → double-free on death →
// freeCount could exceed n → negative alive count + out-of-bounds reads in repro.
//
// seedData layout (filled by gpuSim.paintSeed): [0]=centreX, [1]=centreY,
// [2]=radius, [3]=lineageId, [4]=hue, [5]=initialEnergy, then [6..6+GENOME_SIZE)
// = the genome to clone (selected creature's brain, or a fresh random one).
@group(2) @binding(0) var<storage, read> seedData: array<f32>;

const SEED_COUNT: u32 = 6u;

@compute @workgroup_size(64)
fn seedCreatures(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= SEED_COUNT) { return; }

  // Pop a free slot (lock-free CAS), exactly like the repro pass.
  var slot = NONE;
  loop {
    let cur = atomicLoad(&gridData[freeCountIdx()]);
    if (cur == 0u) { break; }
    let r = atomicCompareExchangeWeak(&gridData[freeCountIdx()], cur, cur - 1u);
    if (r.exchanged) { slot = freeList[cur - 1u]; break; }
  }
  if (slot == NONE) { return; } // ocean full — nothing to seed into

  let frame = P.d1.y;
  let ang = rnd(k + 13u, frame) * TAU;
  let rr = sqrt(rnd(k + 41u, frame)) * seedData[2];
  let sx = wrapf(seedData[0] + cos(ang) * rr, P.p0.x);
  let sy = wrapf(seedData[1] + sin(ang) * rr, P.p0.y);
  state[slot] = vec4<f32>(sx, sy, rnd(k + 7u, frame) * TAU, 0.0);
  bio[slot] = vec4<f32>(seedData[5], seedData[4], 1.0, seedData[3]); // energy, hue, alive, lineage

  let base = 6u;
  for (var j = 0u; j < GENOME_SIZE; j = j + 1u) {
    weights[slot * GENOME_SIZE + j] = seedData[base + j];
  }
}
