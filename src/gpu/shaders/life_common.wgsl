// Shared bindings + helpers for the full GPU life cycle (Phase 2.0).
// Two bind groups keep each under the storage-buffer-per-stage limit.

struct Params {
  p0: vec4<f32>, // worldW, worldH, cellSize, maxSpeed
  p1: vec4<f32>, // maxTurn, dt, eatRadius, foodEnergy
  p2: vec4<f32>, // baseCost, moveCost, reproThreshold, initialEnergy
  p3: vec4<f32>, // mutationRate, mutationStd, offspringFraction, _
  d0: vec4<u32>, // cols, rows, numCells, n
  d1: vec4<u32>, // f (food count), frame, selectedIndex, worldSeed
  ext: vec4<f32>, // predationGain (0 disables), predationMargin, foodPatchiness, bigFoodMult
  ext2: vec4<f32>, // dayNightStrength (0 disables), dayLength (ticks), speciationRate, turnCost
  ext3: vec4<f32>, // bigFoodFraction (>0; idx < f*frac = big food), offspringSpread, glowCost, _
  ext4: vec4<f32>, // reserved (customisation headroom)
  ext5: vec4<f32>, // reserved
  ext6: vec4<f32>, // reserved
};

// Group 0: creature + world state.
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4<f32>>; // x,y,heading,speed
@group(0) @binding(2) var<storage, read_write> bio: array<vec4<f32>>;   // energy, hue, alive, age
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;     // genomes
@group(0) @binding(4) var<storage, read_write> foodPos: array<vec2<f32>>;

// Group 1: grid + lifecycle scratch. To fit maxStorageBuffersPerShaderStage (8),
// everything is packed into the same four buffers; Phase 6 adds a SECOND grid
// over creatures (for neighbour perception + predation) into the SAME buffers so
// no new bindings are needed. Layout of the atomic buffer `gridData`
// (nc = numCells, f = food slots, n = creature slots):
//   [0,    nc)        -> food per-cell count, then reused as scatter cursor
//   [nc,  2nc)        -> creature per-cell count, then reused as scatter cursor
//   [2nc]             -> freeCount (free-slot stack size; persists)
//   [2nc + 1]         -> food spawn budget for this tick
//   [2nc + 2]         -> predation kills this tick (stat; reset each tick)
//   [2nc + 3, +f)     -> per-food eat claim (1 = taken this tick)
//   [2nc+3+f, +n)     -> per-creature predation claim (predatorIndex+1; 0 = none)
//   [2nc+3+f+n]       -> speciation counter (new lineages minted; persists)
//   [2nc+4+f+n, +n)   -> parent lineage id of new lineage k (k = newId - n)
// cellStart holds TWO regions of offsets: food [0, nc] then creatures [nc+1, 2nc+1].
// sortedIdx holds food indices [0, f) then creature indices [f, f+n).
@group(1) @binding(0) var<storage, read_write> gridData: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> cellStart: array<u32>; // 2*(numCells+1)
@group(1) @binding(2) var<storage, read_write> sortedIdx: array<u32>; // food then creatures
@group(1) @binding(3) var<storage, read_write> freeList: array<u32>;  // free slot indices

fn creatureCountIdx(cell: u32) -> u32 { return P.d0.z + cell; } // [numCells, 2numCells)
fn freeCountIdx() -> u32 { return P.d0.z * 2u; }
fn budgetIdx() -> u32 { return P.d0.z * 2u + 1u; }
fn predCountIdx() -> u32 { return P.d0.z * 2u + 2u; }
fn claimIdx(j: u32) -> u32 { return P.d0.z * 2u + 3u + j; }
fn eatenIdx(i: u32) -> u32 { return P.d0.z * 2u + 3u + P.d1.x + i; } // + f + i
fn speciesCountIdx() -> u32 { return P.d0.z * 2u + 3u + P.d1.x + P.d0.w; } // after eatenBy (+f+n)
fn parentIdx(k: u32) -> u32 { return P.d0.z * 2u + 4u + P.d1.x + P.d0.w + k; } // parent of new lineage k
fn creatureStartBase() -> u32 { return P.d0.z + 1u; } // cellStart creature region
fn creatureSortBase() -> u32 { return P.d1.x; }       // sortedIdx creature region (= f)

const WEIGHT_GENES: u32 = 153u; // weights+biases (11 in, 3 out); activation genes follow (one per hidden)
const SIZE_GENE: u32 = 163u; // body-size gene, after the 10 activation genes
const ELONG_GENE: u32 = 164u; // elongation (eel <-> blob)
const FIN_GENE: u32 = 165u; // tail filament (cosmetic)
const GLOW_GENE: u32 = 166u; // bioluminescence brightness
const GENOME_SIZE: u32 = 167u; // WEIGHT_GENES + 10 activation + size + elong + fin + glow
const SIZE_MIN: f32 = 0.6;
const SIZE_MAX: f32 = 2.2;
const NONE: u32 = 0xffffffffu;

// Body-size multiplier of creature i from its size gene (gene 0 -> 1.0).
fn creatureSize(i: u32) -> f32 {
  return clamp(1.0 + 0.5 * weights[i * GENOME_SIZE + SIZE_GENE], SIZE_MIN, SIZE_MAX);
}
// Elongation factor (>1 = streamlined eel, <1 = round blob; gene 0 -> 1.0).
fn creatureElong(i: u32) -> f32 {
  return clamp(1.0 + 0.6 * weights[i * GENOME_SIZE + ELONG_GENE], 0.5, 2.0);
}
// Bioluminescence multiplier (gene 0 -> 1.0).
fn creatureGlow(i: u32) -> f32 {
  return clamp(1.0 + 0.6 * weights[i * GENOME_SIZE + GLOW_GENE], 0.6, 2.0);
}
const TAU: f32 = 6.2831853;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
// Fold the world seed (P.d1.w) into every draw so a seed reproduces not just the
// initial ocean but its whole trajectory (mutation, food respawn). gaussian()
// routes through rnd(), so seeding here covers both.
fn rnd(a: u32, b: u32) -> f32 {
  return f32(pcg(pcg(a ^ P.d1.w) ^ b)) / 4294967296.0;
}
fn gaussian(a: u32, b: u32) -> f32 {
  let u1 = max(rnd(a, b), 1e-7);
  let u2 = rnd(a ^ 0x9e3779b9u, b);
  return sqrt(-2.0 * log(u1)) * cos(TAU * u2);
}

fn wrapf(x: f32, s: f32) -> f32 {
  return x - floor(x / s) * s;
}
fn wrapHue(h: f32) -> f32 {
  return h - floor(h);
}
fn wrapDelta(d: f32, s: f32) -> f32 {
  let h = s * 0.5;
  if (d > h) { return d - s; }
  if (d < -h) { return d + s; }
  return d;
}
fn cellOf(x: f32, y: f32) -> u32 {
  let cs = P.p0.z;
  let cols = P.d0.x;
  let rows = P.d0.y;
  var cx = u32(floor(x / cs));
  var cy = u32(floor(y / cs));
  if (cx >= cols) { cx = cols - 1u; }
  if (cy >= rows) { cy = rows - 1u; }
  return cy * cols + cx;
}
