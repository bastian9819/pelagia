import { initGpu } from './device.js';
import common from './shaders/life_common.wgsl?raw';
import gridPasses from './shaders/life_grid.wgsl?raw';
import simPass from './shaders/life_sim.wgsl?raw';
import cyclePasses from './shaders/life_cycle.wgsl?raw';
import seedPass from './shaders/life_seed.wgsl?raw';
import creatureShader from './shaders/render_quad.wgsl?raw';
import foodShader from './shaders/render_food.wgsl?raw';
import fadeShader from './shaders/fade.wgsl?raw';
import presentShader from './shaders/present.wgsl?raw';
import { DEFAULT_CONFIG } from '../core/config.js';
import {
  GENOME_SIZE,
  WEIGHT_GENES,
  SIZE_GENE,
  ELONG_GENE,
  FIN_GENE,
  GLOW_GENE,
  THERMAL_GENE,
  TOXIN_GENE,
  randomGenome,
} from '../sim/brain.js';
import { pcgHash, floatFromU32, Rng } from '../core/rng.js';
import { SpatialGrid } from '../sim/grid.js';
import { wrapDelta } from '../core/space.js';
import { buildUi, mkBtn, mkIconBtn, setBtnIcon } from './ui.js';
import { icon, type IconName } from './icons.js';
import { buildBrainView } from './brainView.js';
import {
  buildLineagePanel,
  characterizeGenome,
  type LineageRow,
  type LineageTraits,
} from './lineages.js';
import { buildGodPanel, type GodSpec } from './god.js';
import {
  buildObservatory,
  buildEvolutionHistory,
  type ObservatoryData,
  type WorldSample,
  type LineageHistory,
  type Watched,
  type WatchSample,
} from './observatory.js';
import { t, onLang } from './i18n.js';
import { parseShareState, buildShareUrl, applyHash, copyToClipboard } from './share.js';
import inspectShader from './shaders/inspect.wgsl?raw';

/**
 * Phase 2.0: the full evolving ecosystem on the GPU — grid build, sense, brain,
 * move, eat (atomic claim), metabolise, death (free-list) and energy-threshold
 * reproduction into freed slots, plus food respawn. Fixed-capacity buffer (N
 * slots); alive count floats as creatures die and are reborn.
 *
 * Exposes window.__pelagia.readChemotaxis() so emergence can be validated on the
 * GPU the same way as the CPU oracle (mean cos(heading -> nearest food)).
 */
export interface OceanOptions {
  /** Requested creature slot count (capped by GPU memory). */
  n: number;
  /** Sim ticks to run headless before the first rendered frame. */
  warmup?: number;
  /** Called once the ocean is set up and warmed (to dismiss a loading screen). */
  onReady?: () => void;
}

export async function runGpuSim(canvas: HTMLCanvasElement, opts: OceanOptions): Promise<void> {
  // Phase 5: a shared link restores seed + N + god-mode params; a fresh visit
  // gets a random seed so every ocean is novel (and worth sharing). The seed
  // drives both the CPU-side init below and the GPU shader RNG (via pu[23]).
  const share = parseShareState();
  const seed = (share?.seed ?? Math.floor(Math.random() * 0x100000000)) >>> 0;
  const requestedN = share?.n ?? opts.n;
  const { device, format } = await initGpu();
  device.addEventListener('uncapturederror', (e) => {
    // Surface async WebGPU validation errors (otherwise the whole command
    // buffer is silently dropped -> black screen).
    console.error('WebGPU error:', e.error.message);
  });
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('webgpu canvas context unavailable');
  context.configure({ device, format, alphaMode: 'opaque' });

  const cfg = DEFAULT_CONFIG;
  const cellSize = cfg.perceptionRadius;
  const maxN = Math.floor(device.limits.maxStorageBufferBindingSize / (GENOME_SIZE * 4));
  const n = Math.min(requestedN, maxN);
  const f = n; // food slot pool
  // Food must be SCARCE relative to creatures so poor foragers starve -> slots
  // free up -> good foragers reproduce -> selection. Only a fraction start alive.
  const foodInitAlive = Math.round(n * 0.4);
  const world = Math.max(cellSize * 3, 50 * Math.sqrt(n));
  const cols = Math.ceil(world / cellSize);
  const rows = Math.ceil(world / cellSize);
  const numCells = cols * rows;
  // Pheromone field: a fine grid packed into gridData (must match PHERO_RES in
  // life_common.wgsl). 128x128 cells over the world → trails finer than perception.
  const PHERO_RES = 128;
  const PHERO_CELLS = PHERO_RES * PHERO_RES;

  // --- Initial data (seeded: same seed -> same starting ocean) ---
  const stateData = new Float32Array(n * 4);
  const bioData = new Float32Array(n * 4);
  const weightsData = new Float32Array(n * GENOME_SIZE);
  const foodData = new Float32Array(f * 2);
  // Independent RNG streams so positions, genomes and food don't couple, and
  // capping N (low-memory device) still gives the first n creatures the same
  // draws as a higher-N run of the same seed (graceful degradation, R-002).
  // Re-runnable so "restart" can reseed the same ocean in place (no page reload).
  function fillInitialData(): void {
    const rngState = new Rng(seed, 1);
    const rngWeights = new Rng(seed, 2);
    const rngFood = new Rng(seed, 3);
    for (let i = 0; i < n; i++) {
      stateData[i * 4 + 0] = rngState.nextFloat() * world;
      stateData[i * 4 + 1] = rngState.nextFloat() * world;
      stateData[i * 4 + 2] = rngState.nextFloat() * Math.PI * 2;
      stateData[i * 4 + 3] = 0; // speed (reset on restart)
      bioData[i * 4 + 0] = cfg.initialEnergy;
      bioData[i * 4 + 1] = floatFromU32(pcgHash(i)); // hue derived from lineage id
      bioData[i * 4 + 2] = 1; // alive
      bioData[i * 4 + 3] = i; // lineage id (= founder slot; inherited unchanged)
    }
    for (let i = 0; i < n; i++) {
      const wb = i * GENOME_SIZE;
      for (let k = 0; k < WEIGHT_GENES; k++) {
        weightsData[wb + k] = (rngWeights.nextFloat() * 2 - 1) * cfg.weightInitStd;
      }
      // Activation genes biased active (uniform(-0.3, 1.0), ~77% on) so brains
      // start capable; mutation then evolves how many neurons each lineage uses.
      for (let k = WEIGHT_GENES; k < SIZE_GENE; k++) {
        weightsData[wb + k] = rngWeights.nextFloat() * 1.3 - 0.3;
      }
      // Body-size gene with real spread (~N(1, 0.4) bodies) so predator/prey size
      // niches can form from the start instead of waiting for variance to build.
      weightsData[wb + SIZE_GENE] = rngWeights.nextGaussian() * 0.8;
      // Morphology genes seeded with spread so creatures look varied from tick 0:
      // elongation (eel↔blob), tail filament and bioluminescence all evolve.
      weightsData[wb + ELONG_GENE] = rngWeights.nextGaussian() * 0.7;
      weightsData[wb + FIN_GENE] = rngWeights.nextGaussian() * 0.9;
      weightsData[wb + GLOW_GENE] = rngWeights.nextGaussian() * 0.7;
      weightsData[wb + THERMAL_GENE] = rngWeights.nextGaussian() * 0.6; // thermal preference spread
      // Toxicity: biased low (most creatures harmless; a toxic minority can evolve).
      weightsData[wb + TOXIN_GENE] = rngWeights.nextGaussian() * 0.35 - 0.15;
    }
    for (let j = 0; j < f; j++) {
      if (j < foodInitAlive) {
        foodData[j * 2 + 0] = rngFood.nextFloat() * world;
        foodData[j * 2 + 1] = rngFood.nextFloat() * world;
      } else {
        foodData[j * 2 + 0] = -1; // dead slot (sentinel)
        foodData[j * 2 + 1] = -1;
      }
    }
  }
  fillInitialData();

  // --- Buffers ---
  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const CS = GPUBufferUsage.COPY_SRC;
  const stateBuf = device.createBuffer({ size: stateData.byteLength, usage: S | CD | CS });
  const bioBuf = device.createBuffer({ size: bioData.byteLength, usage: S | CD | CS });
  const weightsBuf = device.createBuffer({ size: weightsData.byteLength, usage: S | CD | CS });
  const foodBuf = device.createBuffer({ size: foodData.byteLength, usage: S | CD | CS });
  // Phase 6: gridData / cellStart / sortedIdx hold BOTH the food grid and a
  // second CREATURE grid (neighbour perception + predation), packed into the same
  // buffers to stay under the 8-storage-buffer limit (see life_common.wgsl).
  // gridData also carries freeCount, food budget, the predation counter, food
  // claims and per-creature predation claims. COPY_DST so restart can reset
  // freeCount in place.
  // ... + speciation counter (1) + parent-lineage pointers (n) for the family tree.
  const gridDataBuf = device.createBuffer({
    // ... + a PHERO_CELLS pheromone field at the very end (pheroBase in shaders).
    size: (2 * numCells + 4 + f + 2 * n + PHERO_CELLS) * 4,
    usage: S | CS | CD,
  });
  const speciesCountOffset = (2 * numCells + 3 + f + n) * 4; // byte offset of the counter
  const cellStartBuf = device.createBuffer({ size: 2 * (numCells + 1) * 4, usage: S });
  const sortedBuf = device.createBuffer({ size: (f + n) * 4, usage: S });
  const freeListBuf = device.createBuffer({ size: n * 4, usage: S });
  // Brain-inspector output (one selected creature): inputs(13)|hidden(10)|
  // outputs(3)|state(8)|activeCount|size|elong|glow|thermal = 39 floats; padded.
  const INSPECT_FLOATS = 44;
  const inspectBuf = device.createBuffer({ size: INSPECT_FLOATS * 4, usage: S | CS });
  const inspectReadback = device.createBuffer({
    size: INSPECT_FLOATS * 4,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  // Watch-list read-back: a few tracked creatures sampled through the inspect pass.
  const MAX_WATCH = 8;
  const watchReadback = device.createBuffer({
    size: MAX_WATCH * INSPECT_FLOATS * 4,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  device.queue.writeBuffer(stateBuf, 0, stateData);
  device.queue.writeBuffer(bioBuf, 0, bioData);
  device.queue.writeBuffer(weightsBuf, 0, weightsData);
  device.queue.writeBuffer(foodBuf, 0, foodData);

  // --- Params uniform (192 bytes: 10 vec4<f32> + 2 vec4<u32>) ---
  // Appended additively so existing indices never move: d0 = pu[16..19],
  // d1 = pu[20..23], ext = pf[24..27], ext2 = pf[28..31] (day/night + turnCost),
  // ext3..ext6 = pf[32..47] (customisation headroom). Grown from 128B (gotcha #5).
  const paramsBuf = device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | CD });
  const pbuf = new ArrayBuffer(192);
  const pf = new Float32Array(pbuf);
  const pu = new Uint32Array(pbuf);
  pf[0] = world;
  pf[1] = world;
  pf[2] = cellSize;
  pf[3] = cfg.maxSpeed;
  pf[4] = cfg.maxTurnRate;
  pf[5] = 1; // dt
  pf[6] = cfg.eatRadius;
  pf[7] = cfg.foodEnergy;
  pf[8] = cfg.baseCost;
  pf[9] = cfg.moveCost;
  pf[10] = cfg.reproductionThreshold;
  pf[11] = cfg.initialEnergy;
  pf[12] = cfg.mutationRate;
  pf[13] = cfg.mutationStd;
  pf[14] = cfg.offspringEnergyFraction;
  // Food spawn budget per tick. Caps energy influx so carrying capacity sits
  // well below N -> heavy death/birth turnover -> strong selection. Matched to
  // the CPU ecology's food-influx density; calibrated via alive + chemotaxis.
  pf[15] = Math.max(1, Math.round(n * 0.005));
  pu[16] = cols;
  pu[17] = rows;
  pu[18] = numCells;
  pu[19] = n;
  pu[20] = f;
  pu[23] = seed >>> 0; // world seed: folded into the shader RNG (mutation/respawn)
  // Predation gain (fraction of prey energy the predator gains; 0 disables it) and
  // predation margin (how much bigger you must be to eat another, >= 1). Both are
  // god-mode sliders, serialised in the share URL. Moderate defaults = food stays
  // the primary resource, predation a secondary pressure (anti-gray-soup, R-001).
  pf[24] = 0.5;
  pf[25] = 1.25;
  // Food patchiness (ext.z): 0 = uniform sprinkle, 1 = tight drifting blooms.
  pf[26] = 0.6;
  // Big-food value (ext.w): energy multiplier of a rare big-food pellet vs plankton.
  pf[27] = 5;
  // Day/night cycle: ext2.x = strength (0 disables, food + light swing), ext2.y =
  // period in ticks. On by default (moderate) so the boom/bust is visible.
  pf[28] = 0.45;
  pf[29] = 1600;
  // Speciation rate (ext2.z): chance per birth that the offspring founds a new
  // lineage (new colour + parent link) → a branching family tree over time.
  pf[30] = 0.004;
  // ext2.w turnCost: energy spent per unit of |turn| (0 = turning is free).
  pf[31] = 0.0;
  // ext3.x bigFoodFraction: pellets with index < f*frac are "big food" (default
  // 1/16 ≈ 0.0625, matching the old hardcoded f/16); 0 disables big food.
  pf[32] = 0.0625;
  // ext3.y offspringSpread: how far a newborn is jittered from its parent (world
  // units; default 4 = the old hardcoded value). ext3.z glowCost: metabolic cost
  // of bioluminescence per unit of glow above 1 (0 = free).
  pf[33] = 4.0;
  pf[34] = 0.0;
  // ext3.w currentStrength: ocean current that advects creatures (0 = still water).
  pf[35] = 0.4;
  // ext4.x attackCost: energy spent each tick the brain's attack output is on, so
  // indiscriminate aggression is selected against (default small but non-zero).
  pf[36] = 0.04;
  // ext4.y thermalContrast: how strongly a temperature mismatch costs energy
  // (0 = uniform ocean / no biomes). Moderate default so biomes matter but don't
  // wipe out mismatched zones (creatures can also swim toward their comfort band).
  pf[37] = 0.05;
  // ext4.z carrionAmount: chance a dead creature drops a food pellet where it fell
  // (0 = off). Feeds scavengers; bounded by one pellet per food slot.
  pf[38] = 0.3;
  // ext4.w toxinPotency: energy a predator loses per unit of prey toxicity (0 = off).
  pf[39] = 15;
  // ext6.y sexualRate: chance a birth is recombined from two parents (crossover)
  // instead of a clone (0 = fully asexual). Mixes traits across nearby lineages.
  pf[45] = 0.25;
  // ext6.z pheroDeposit: pheromone units each creature lays per tick (0 = off).
  pf[46] = 256;
  // ext6.w mateChoice: strength of sexual selection on the glow ornament during
  // sexual reproduction. 0 = mate with the nearest neighbour (no choice); higher =
  // prefer brighter mates, so the glow gene trends up over generations (runaway).
  // Only affects the fraction of births that are sexual (ext6.y). See D-039.
  pf[47] = 1.0;
  function writeParams(frame: number): void {
    pu[21] = frame;
    device.queue.writeBuffer(paramsBuf, 0, pbuf);
  }

  // --- Bind groups (two, to stay under the per-stage storage limit) ---
  const u = GPUShaderStage.COMPUTE;
  const sto: GPUBufferBindingLayout = { type: 'storage' };
  const bgl0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: u, buffer: { type: 'uniform' } },
      { binding: 1, visibility: u, buffer: sto },
      { binding: 2, visibility: u, buffer: sto },
      { binding: 3, visibility: u, buffer: sto },
      { binding: 4, visibility: u, buffer: sto },
    ],
  });
  const bgl1 = device.createBindGroupLayout({
    entries: [0, 1, 2, 3].map((binding) => ({ binding, visibility: u, buffer: sto })),
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
  const group0 = device.createBindGroup({
    layout: bgl0,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } },
      { binding: 4, resource: { buffer: foodBuf } },
    ],
  });
  const group1 = device.createBindGroup({
    layout: bgl1,
    entries: [
      { binding: 0, resource: { buffer: gridDataBuf } },
      { binding: 1, resource: { buffer: cellStartBuf } },
      { binding: 2, resource: { buffer: sortedBuf } },
      { binding: 3, resource: { buffer: freeListBuf } },
    ],
  });

  const gridModule = device.createShaderModule({ code: common + '\n' + gridPasses });
  const simModule = device.createShaderModule({ code: common + '\n' + simPass });
  const cycleModule = device.createShaderModule({ code: common + '\n' + cyclePasses });
  const seedModule = device.createShaderModule({ code: common + '\n' + seedPass });
  const wgslErrors: string[] = [];
  for (const [name, mod] of [
    ['grid', gridModule],
    ['sim', simModule],
    ['cycle', cycleModule],
    ['seed', seedModule],
  ] as const) {
    const info = await mod.getCompilationInfo();
    for (const m of info.messages) {
      if (m.type === 'error')
        wgslErrors.push(`[WGSL ${name}] ${m.lineNum}:${m.linePos} ${m.message}`);
    }
  }
  if (wgslErrors.length) throw new Error('Shader compile errors:\n' + wgslErrors.join('\n'));
  const mk = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
    device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
  const pGridClear = mk(gridModule, 'gridClear');
  const pClaimClear = mk(gridModule, 'claimClear');
  const pCount = mk(gridModule, 'count');
  const pCountCreatures = mk(gridModule, 'countCreatures');
  const pScan = mk(gridModule, 'scan');
  const pScatter = mk(gridModule, 'scatter');
  const pScatterCreatures = mk(gridModule, 'scatterCreatures');
  const pSim = mk(simModule, 'main');
  const pDeath = mk(cycleModule, 'death');
  const pRepro = mk(cycleModule, 'repro');
  const pRespawn = mk(cycleModule, 'foodRespawn');
  const pPheroDecay = mk(gridModule, 'pheroDecay');

  // Brain inspector: its own bind group (uniform + 6 read-only + 1 storage = 7).
  const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
  const bglInspect = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: u, buffer: { type: 'uniform' } },
      { binding: 1, visibility: u, buffer: ro },
      { binding: 2, visibility: u, buffer: ro },
      { binding: 3, visibility: u, buffer: ro },
      { binding: 4, visibility: u, buffer: ro },
      { binding: 5, visibility: u, buffer: ro },
      { binding: 6, visibility: u, buffer: ro },
      { binding: 7, visibility: u, buffer: sto },
      { binding: 8, visibility: u, buffer: ro }, // gridData (pheromone field, read-only)
    ],
  });
  const inspectPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglInspect] }),
    compute: { module: device.createShaderModule({ code: inspectShader }), entryPoint: 'main' },
  });
  const inspectBind = device.createBindGroup({
    layout: bglInspect,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } },
      { binding: 4, resource: { buffer: foodBuf } },
      { binding: 5, resource: { buffer: cellStartBuf } },
      { binding: 6, resource: { buffer: sortedBuf } },
      { binding: 7, resource: { buffer: inspectBuf } },
      { binding: 8, resource: { buffer: gridDataBuf } },
    ],
  });

  // Seed brush pass: pops free slots and writes new creatures (correct free-list
  // handling, unlike a CPU write-by-index). seedData carries centre/radius/lineage/
  // hue/energy + the genome to clone (6 + GENOME_SIZE floats). The pass only touches
  // state/bio/weights + gridData/freeList + seedData, so it uses TRIMMED bind-group
  // layouts (not the full sim ones) to stay within the 8-storage-buffer limit
  // (gotcha #1): reusing the sim's bgl0+bgl1 would bind 8 buffers, +seedData = 9.
  const seedDataBuf = device.createBuffer({ size: (6 + GENOME_SIZE) * 4, usage: S | CD });
  const bglSeed0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: u, buffer: { type: 'uniform' } }, // P
      { binding: 1, visibility: u, buffer: sto }, // state
      { binding: 2, visibility: u, buffer: sto }, // bio
      { binding: 3, visibility: u, buffer: sto }, // weights
    ],
  });
  const bglSeed1 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: u, buffer: sto }, // gridData (binding numbers match life_common)
      { binding: 3, visibility: u, buffer: sto }, // freeList
    ],
  });
  const bglSeed2 = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: u, buffer: ro }], // seedData
  });
  const pSeed = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bglSeed0, bglSeed1, bglSeed2] }),
    compute: { module: seedModule, entryPoint: 'seedCreatures' },
  });
  const seedGroup0 = device.createBindGroup({
    layout: bglSeed0,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } },
    ],
  });
  const seedGroup1 = device.createBindGroup({
    layout: bglSeed1,
    entries: [
      { binding: 0, resource: { buffer: gridDataBuf } },
      { binding: 3, resource: { buffer: freeListBuf } },
    ],
  });
  const seedBind = device.createBindGroup({
    layout: bglSeed2,
    entries: [{ binding: 0, resource: { buffer: seedDataBuf } }],
  });

  const wgN = Math.ceil(n / 256);
  const wgF = Math.ceil(f / 256);
  const wgC2 = Math.ceil((2 * numCells) / 256); // clear both grids' cell counts
  function recordSim(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, group0);
    pass.setBindGroup(1, group1);
    pass.setPipeline(pGridClear);
    pass.dispatchWorkgroups(wgC2);
    pass.setPipeline(pClaimClear);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pCount);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pCountCreatures);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pScan);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pScatter);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pScatterCreatures);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pSim);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pDeath);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pRepro);
    pass.dispatchWorkgroups(wgN);
    pass.setPipeline(pRespawn);
    pass.dispatchWorkgroups(wgF);
    pass.setPipeline(pPheroDecay);
    pass.dispatchWorkgroups(Math.ceil(PHERO_CELLS / 256));
    pass.end();
  }

  // --- Render: HDR accumulation texture -> trails + glow -> tonemapped present ---
  const renderUbo = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | CD });
  const renderData = new Float32Array(20);
  renderData[11] = world; // world size (for the present pass's thermal-field tint)
  renderData[16] = 2 * numCells + 4 + f + 2 * n; // pheroBase (u32 index into gridData)
  let fieldTint = 0; // 0 = off; the "fields" toggle sets a faint tint strength
  let lockOn = false; // camera follows the selected creature when on
  let currentViz = false; // draw animated current streaks when on
  let pheroViz = false; // draw the pheromone field (trails) when on
  renderData[7] = Math.floor(f * pf[32]); // big-food slot count (tracks g_bigFoodAmt)
  // renderData[8] = highlighted lineage id, [9] = highlight on (1/0).
  let highlightOn = false;
  function applyHighlight(): void {
    renderData[8] = selectedLineage;
    renderData[9] = highlightOn && selectedIndex >= 0 ? 1 : 0;
  }
  const ACCUM_FORMAT: GPUTextureFormat = 'rgba16float';
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const additive: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  };
  const overBlend: GPUBlendState = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };
  const mkRender = (
    code: string,
    blend: GPUBlendState | undefined,
    target: GPUTextureFormat,
  ): GPURenderPipeline => {
    const mod = device.createShaderModule({ code });
    const colorTarget: GPUColorTargetState = blend ? { format: target, blend } : { format: target };
    return device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [colorTarget] },
      primitive: { topology: 'triangle-list' },
    });
  };
  const creaturePipeline = mkRender(creatureShader, additive, ACCUM_FORMAT);
  const foodPipeline = mkRender(foodShader, additive, ACCUM_FORMAT);
  const fadePipeline = mkRender(fadeShader, overBlend, ACCUM_FORMAT);
  const presentPipeline = mkRender(presentShader, undefined, format);

  const creatureBind = device.createBindGroup({
    layout: creaturePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUbo } },
      { binding: 1, resource: { buffer: stateBuf } },
      { binding: 2, resource: { buffer: bioBuf } },
      { binding: 3, resource: { buffer: weightsBuf } }, // body-size gene for sizing
    ],
  });
  const foodBind = device.createBindGroup({
    layout: foodPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUbo } },
      { binding: 1, resource: { buffer: foodBuf } },
    ],
  });

  let accumTex: GPUTexture | undefined;
  let accumView!: GPUTextureView;
  let presentBind!: GPUBindGroup;
  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.floor(window.innerWidth * dpr));
    const ch = Math.max(1, Math.floor(window.innerHeight * dpr));
    canvas.width = cw;
    canvas.height = ch;

    accumTex?.destroy();
    accumTex = device.createTexture({
      size: [cw, ch],
      format: ACCUM_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    accumView = accumTex.createView();
    presentBind = device.createBindGroup({
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: accumView },
        { binding: 2, resource: { buffer: renderUbo } },
        { binding: 3, resource: { buffer: gridDataBuf } },
      ],
    });
    // Clear the fresh accumulation texture once.
    const enc = device.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: accumView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      .end();
    device.queue.submit([enc.finish()]);
    updateView();
  }

  // --- Camera (zoom + pan) ---
  const cam = { zoom: 1, cx: world / 2, cy: world / 2 };
  function ppwNow(): number {
    return Math.min(canvas.width / world, canvas.height / world) * cam.zoom;
  }
  function clampCam(): void {
    cam.zoom = Math.min(60, Math.max(1, cam.zoom));
    cam.cx = Math.min(world, Math.max(0, cam.cx));
    cam.cy = Math.min(world, Math.max(0, cam.cy));
  }
  function updateView(): void {
    const cw = canvas.width;
    const ch = canvas.height;
    const ppw = ppwNow();
    renderData[0] = (2 * ppw) / cw; // sx (world -> NDC)
    renderData[2] = (-2 * cam.cx * ppw) / cw; // ox
    renderData[1] = (-2 * ppw) / ch; // sy (flip y)
    renderData[3] = (2 * cam.cy * ppw) / ch; // oy
    // Glow radius in world units, set so the ON-SCREEN size stays in a sane band
    // (~2.5px when zoomed out so creatures are still visible, capped ~16px when
    // zoomed in so they read as creatures, not giant blobs). 5 = creature scale.
    renderData[4] = Math.min(16 / ppw, Math.max(2.5 / ppw, 5));
    renderData[5] = 1.4; // brightness (HDR; tonemapped on present)
    device.queue.writeBuffer(renderUbo, 0, renderData);
  }
  function screenToWorld(clientX: number, clientY: number): { wx: number; wy: number } {
    const dpr = canvas.width / window.innerWidth;
    const ppw = ppwNow();
    return {
      wx: (clientX * dpr - canvas.width / 2) / ppw + cam.cx,
      wy: (clientY * dpr - canvas.height / 2) / ppw + cam.cy,
    };
  }
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const before = screenToWorld(e.clientX, e.clientY);
      cam.zoom *= Math.exp(-e.deltaY * 0.0015);
      clampCam();
      const after = screenToWorld(e.clientX, e.clientY);
      cam.cx += before.wx - after.wx; // keep the point under the cursor fixed
      cam.cy += before.wy - after.wy;
      clampCam();
      updateView();
    },
    { passive: false },
  );

  // --- Interactive brush ("hand of god"): with a tool selected, dragging over the
  // ocean attracts/repels/feeds/smites creatures instead of panning. tool 0 = none
  // (normal pan + select). The force tools drive ext5/ext6 in the sim shader; the
  // food tool scatters pellets into a rolling block of slots around the cursor. ---
  const BRUSH = { tool: 0, radius: 130, strength: 6 };
  // none, attract, repel, food(CPU), cataclysm, seed(CPU), mutagen, heal.
  // CPU tools (food, seed) = shader mode 0.
  const SHADER_MODE = [0, 1, 2, 0, 4, 0, 6, 7];
  let painting = false;
  let brushWX = 0;
  let brushWY = 0;
  let foodCursor = 0;
  const FOOD_BRUSH_COUNT = 32;
  const foodBrushData = new Float32Array(FOOD_BRUSH_COUNT * 2);
  const wrapWorld = (v: number): number => ((v % world) + world) % world;
  function writeBrushParams(): void {
    pf[40] = brushWX;
    pf[41] = brushWY;
    pf[42] = painting && BRUSH.tool !== 3 ? SHADER_MODE[BRUSH.tool]! : 0;
    pf[43] = BRUSH.radius;
    pf[44] = BRUSH.strength;
    device.queue.writeBuffer(paramsBuf, 0, pbuf);
  }
  function paintFood(): void {
    for (let k = 0; k < FOOD_BRUSH_COUNT; k++) {
      const a = Math.random() * Math.PI * 2;
      const rr = Math.sqrt(Math.random()) * BRUSH.radius;
      foodBrushData[k * 2] = wrapWorld(brushWX + Math.cos(a) * rr);
      foodBrushData[k * 2 + 1] = wrapWorld(brushWY + Math.sin(a) * rr);
    }
    const start = foodCursor % f;
    const count = Math.min(FOOD_BRUSH_COUNT, f - start);
    device.queue.writeBuffer(foodBuf, start * 8, foodBrushData, 0, count * 2);
    foodCursor = (foodCursor + count) % f;
  }

  // Seed brush ("inseminate"): drop new creatures around the cursor into a rolling
  // block of slots. Clones the SELECTED creature's brain if one is selected (spread
  // its lineage), else a fresh random brain in a brand-new lineage. Writes state +
  // bio + genome directly (a god tool; seeded creatures may be ephemeral if the GPU
  // later reclaims a reactivated free slot — acceptable for a sandbox).
  let seedGenome: Float32Array | null = null;
  // seedData packs the seed pass's inputs: centre x/y, radius, lineage, hue, energy,
  // then the genome to clone. The GPU pass pops free slots (correct accounting).
  const seedData = new Float32Array(6 + GENOME_SIZE);
  const strokeGenome = new Float32Array(GENOME_SIZE);
  let seedLineageCounter = 0;
  let strokeLineage = 0;
  let strokeHue = 0;
  function beginSeedStroke(): void {
    if (seedGenome) {
      strokeGenome.set(seedGenome);
      strokeLineage = selectedLineage; // clones join the selected lineage
    } else {
      const rng = new Rng((Math.random() * 0xffffffff) >>> 0, 7);
      randomGenome(strokeGenome, 0, rng, cfg.weightInitStd);
      strokeLineage = 2_000_000 + seedLineageCounter++; // fresh clade, no id collision
    }
    strokeHue = floatFromU32(pcgHash(strokeLineage >>> 0));
  }
  function paintSeed(): void {
    seedData[0] = brushWX;
    seedData[1] = brushWY;
    seedData[2] = BRUSH.radius;
    seedData[3] = strokeLineage;
    seedData[4] = strokeHue;
    seedData[5] = pf[11]!; // initial energy
    seedData.set(strokeGenome, 6);
    device.queue.writeBuffer(seedDataBuf, 0, seedData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, seedGroup0);
    pass.setBindGroup(1, seedGroup1);
    pass.setBindGroup(2, seedBind);
    pass.setPipeline(pSeed);
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  let dragging = false;
  let dragMoved = false;
  let lastX = 0;
  let lastY = 0;

  // Brush area-of-effect ring: a circle under the cursor showing exactly where (and
  // how big) the active brush acts. Without it, attract/repel/mutate looked like
  // they did nothing — you couldn't see the zone they affect, especially zoomed out.
  const brushRing = document.createElement('div');
  brushRing.style.cssText =
    'position:fixed;display:none;border:1.5px dashed rgba(63,240,216,0.8);border-radius:50%;' +
    'pointer-events:none;transform:translate(-50%,-50%);z-index:5;mix-blend-mode:screen;';
  document.body.appendChild(brushRing);
  function updateBrushRing(clientX: number, clientY: number): void {
    if (BRUSH.tool === 0) {
      brushRing.style.display = 'none';
      return;
    }
    const dpr = canvas.width / window.innerWidth;
    const d = (2 * BRUSH.radius * ppwNow()) / dpr; // world radius -> on-screen diameter
    brushRing.style.left = `${clientX}px`;
    brushRing.style.top = `${clientY}px`;
    brushRing.style.width = `${d}px`;
    brushRing.style.height = `${d}px`;
    brushRing.style.display = 'block';
  }

  canvas.addEventListener('pointerleave', () => (brushRing.style.display = 'none'));
  canvas.addEventListener('pointerdown', (e) => {
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      /* ignore (e.g. synthetic events) */
    }
    if (BRUSH.tool !== 0) {
      painting = true;
      const { wx, wy } = screenToWorld(e.clientX, e.clientY);
      brushWX = wx;
      brushWY = wy;
      updateBrushRing(e.clientX, e.clientY);
      writeBrushParams();
      if (BRUSH.tool === 3) paintFood();
      else if (BRUSH.tool === 5) {
        beginSeedStroke();
        paintSeed();
      }
      return;
    }
    dragging = true;
    dragMoved = false;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (BRUSH.tool !== 0) updateBrushRing(e.clientX, e.clientY); // preview the area
    if (painting) {
      const { wx, wy } = screenToWorld(e.clientX, e.clientY);
      brushWX = wx;
      brushWY = wy;
      writeBrushParams();
      if (BRUSH.tool === 3) paintFood();
      else if (BRUSH.tool === 5) paintSeed();
      return;
    }
    if (!dragging) return;
    if (Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY) > 4) dragMoved = true;
    const ppw = ppwNow();
    const dpr = canvas.width / window.innerWidth;
    cam.cx -= ((e.clientX - lastX) * dpr) / ppw;
    cam.cy -= ((e.clientY - lastY) * dpr) / ppw;
    lastX = e.clientX;
    lastY = e.clientY;
    clampCam();
    updateView();
  });
  canvas.addEventListener('pointerup', (e) => {
    if (painting) {
      painting = false;
      writeBrushParams(); // mode -> 0 (forces stop)
      return;
    }
    dragging = false;
    if (!dragMoved) {
      const { wx, wy } = screenToWorld(e.clientX, e.clientY);
      void selectNear(wx, wy);
    }
  });

  resize();
  window.addEventListener('resize', resize);

  // --- Brush toolbar (bottom-left): pick a tool, then drag over the ocean. ---
  const brushBar = document.createElement('div');
  brushBar.className = 'pg-panel';
  brushBar.style.cssText =
    'position:fixed;left:14px;bottom:18px;display:flex;gap:4px;align-items:center;z-index:6;' +
    'padding:7px;border-radius:14px;';
  const brushTools: { tool: number; icon: IconName; key: string }[] = [
    { tool: 0, icon: 'move', key: 'tool_pan' },
    { tool: 1, icon: 'magnet', key: 'tool_attract' },
    { tool: 2, icon: 'wind', key: 'tool_repel' },
    { tool: 3, icon: 'droplets', key: 'tool_food' },
    { tool: 7, icon: 'heart', key: 'tool_heal' },
    { tool: 5, icon: 'sprout', key: 'tool_seed' },
    { tool: 6, icon: 'flask', key: 'tool_mutagen' },
    { tool: 4, icon: 'flame', key: 'tool_smite' },
  ];
  const brushBtns: { b: HTMLButtonElement; def: (typeof brushTools)[number] }[] = [];
  function setTool(tool: number): void {
    BRUSH.tool = tool;
    canvas.style.cursor = tool === 0 ? '' : 'crosshair';
    if (tool === 0) {
      brushRing.style.display = 'none';
      if (painting) {
        painting = false;
        writeBrushParams();
      }
    }
    for (const { b, def } of brushBtns) b.classList.toggle('is-active', def.tool === tool);
  }
  for (const def of brushTools) {
    const b = mkIconBtn(def.icon, t(def.key), () => setTool(def.tool));
    brushBtns.push({ b, def });
    brushBar.append(b);
  }
  const brushSep = document.createElement('span');
  brushSep.style.cssText = 'width:1px;height:20px;background:var(--border-1);margin:0 3px;';
  brushBar.append(brushSep);
  const sizeInput = document.createElement('input');
  sizeInput.type = 'range';
  sizeInput.className = 'pg-range';
  sizeInput.min = '40';
  sizeInput.max = '300';
  sizeInput.step = '10';
  sizeInput.value = String(BRUSH.radius);
  sizeInput.title = t('tool_size');
  sizeInput.style.cssText = 'width:70px;margin:0 4px;';
  sizeInput.addEventListener('input', () => {
    BRUSH.radius = Number(sizeInput.value);
    if (painting) writeBrushParams();
    if (brushRing.style.display !== 'none') {
      const dpr = canvas.width / window.innerWidth;
      const d = (2 * BRUSH.radius * ppwNow()) / dpr;
      brushRing.style.width = `${d}px`;
      brushRing.style.height = `${d}px`;
    }
  });
  brushBar.append(sizeInput);
  document.body.appendChild(brushBar);
  setTool(0);
  onLang(() => {
    for (const { b, def } of brushBtns) b.title = t(def.key);
    sizeInput.title = t('tool_size');
  });

  // Photo mode: press H to hide every overlay (all body children but the canvas)
  // for a clean screenshot; press again to restore.
  let photoMode = false;
  const photoSaved: { el: HTMLElement; display: string }[] = [];
  window.addEventListener('keydown', (e) => {
    if (e.code !== 'KeyH' || e.metaKey || e.ctrlKey || e.altKey) return;
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) return;
    photoMode = !photoMode;
    if (photoMode) {
      for (const el of Array.from(document.body.children)) {
        if (el === canvas || !(el instanceof HTMLElement) || el.style.display === 'none') continue;
        photoSaved.push({ el, display: el.style.display });
        el.style.display = 'none';
      }
    } else {
      for (const { el, display } of photoSaved) el.style.display = display;
      photoSaved.length = 0;
    }
  });

  // --- Colour-by-trait: how creatures are tinted (renderData[6]); button lives
  // on the transport bar and is driven through this control. ---
  const COLOR_MODES = [
    'lineageWord',
    'sizeWord',
    'neurons',
    'energyWord',
    'speedWord',
    'elongWord',
    'glowWord',
    'thermalWord',
    'toxinWord',
  ];
  let colorMode = 0;

  // --- Stats panel + controls ---
  const ui = buildUi(
    () => {
      cam.zoom = 1;
      cam.cx = world / 2;
      cam.cy = world / 2;
      updateView();
    },
    () => {
      // Step one tick (used while paused) so a single decision can be studied.
      writeParams(frame);
      const enc = device.createCommandEncoder();
      recordSim(enc);
      device.queue.submit([enc.finish()]);
      frame++;
    },
    {
      cycle: () => {
        colorMode = (colorMode + 1) % COLOR_MODES.length;
        renderData[6] = colorMode;
        device.queue.writeBuffer(renderUbo, 0, renderData);
      },
      label: () => t(COLOR_MODES[colorMode]!),
    },
  );
  document.body.appendChild(ui.panel);
  document.body.appendChild(ui.controls);
  document.body.appendChild(ui.menu);

  // --- Selection + brain inspector ---
  let selectedIndex = -1;
  let selectedLineage = -1; // to detect slot reuse (a different creature took the slot)
  let selecting = false;
  let inspectPending = false;
  // Last sample while the selected creature was alive — so that when it dies we can
  // freeze the panel on its final state even if a newborn has already taken its slot.
  let lastGoodSample: Float32Array | null = null;
  const brainView = buildBrainView(
    () => deselect(),
    () => {
      if (selectedIndex >= 0) addWatch(selectedIndex, selectedLineage);
    },
  );
  document.body.appendChild(brainView.panel);
  const ring = document.createElement('div');
  ring.style.cssText =
    'position:fixed;display:none;border:2px solid rgba(63,240,216,0.9);border-radius:50%;' +
    'box-shadow:0 0 12px rgba(63,240,216,0.6);pointer-events:none;transform:translate(-50%,-50%);z-index:5;';
  document.body.appendChild(ring);

  function setSelectedUniform(): void {
    pu[22] = selectedIndex >= 0 ? selectedIndex : 0xffffffff;
  }
  function deselect(): void {
    selectedIndex = -1;
    selectedLineage = -1;
    lastGoodSample = null;
    setSelectedUniform();
    applyHighlight();
    brainView.hide();
    brainView.setGenome(null);
    seedGenome = null; // nothing to clone with the seed brush
    ring.style.display = 'none';
  }
  // Read one creature's genome back (static per creature) for the policy view.
  async function loadGenome(index: number, lineage: number): Promise<void> {
    const gRead = device.createBuffer({
      size: GENOME_SIZE * 4,
      usage: GPUBufferUsage.MAP_READ | CD,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(weightsBuf, index * GENOME_SIZE * 4, gRead, 0, GENOME_SIZE * 4);
    device.queue.submit([enc.finish()]);
    await gRead.mapAsync(GPUMapMode.READ);
    const g = new Float32Array(gRead.getMappedRange().slice(0));
    gRead.unmap();
    gRead.destroy();
    if (selectedIndex === index && selectedLineage === lineage) {
      brainView.setGenome(g);
      seedGenome = g; // the seed brush can clone this brain ("inseminate")
    }
  }
  function positionRing(d: Float32Array): void {
    const ppw = ppwNow();
    const dpr = canvas.width / window.innerWidth;
    ring.style.left = `${((d[30]! - cam.cx) * ppw + canvas.width / 2) / dpr}px`;
    ring.style.top = `${((d[31]! - cam.cy) * ppw + canvas.height / 2) / dpr}px`;
    const px = Math.max(18, (renderData[4]! * ppw * 2) / dpr + 10);
    ring.style.width = `${px}px`;
    ring.style.height = `${px}px`;
    ring.style.display = 'block';
  }
  async function selectNear(wx: number, wy: number): Promise<void> {
    if (selecting) return;
    selecting = true;
    const sRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(stateBuf, 0, sRead, 0, n * 16);
    enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
    device.queue.submit([enc.finish()]);
    await Promise.all([sRead.mapAsync(GPUMapMode.READ), bRead.mapAsync(GPUMapMode.READ)]);
    const st = new Float32Array(sRead.getMappedRange().slice(0));
    const bi = new Float32Array(bRead.getMappedRange().slice(0));
    sRead.unmap();
    bRead.unmap();
    sRead.destroy();
    bRead.destroy();

    const maxDist = 26 / ppwNow();
    let best = -1;
    let bestD2 = maxDist * maxDist;
    for (let i = 0; i < n; i++) {
      if (bi[i * 4 + 2]! < 0.5) continue;
      const dx = st[i * 4]! - wx;
      const dy = st[i * 4 + 1]! - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = i;
      }
    }
    selecting = false;
    if (best >= 0) {
      selectedIndex = best;
      selectedLineage = Math.round(bi[best * 4 + 3]!);
      lastGoodSample = null; // fresh selection; captured on the first live frame
      setSelectedUniform();
      applyHighlight();
      brainView.show();
      void loadGenome(best, selectedLineage);
    } else {
      deselect();
    }
  }
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') deselect();
  });

  // --- Lineage explorer ---
  const lineagePanel = buildLineagePanel();
  document.body.appendChild(lineagePanel.panel);
  ui.addTool(lineagePanel.toggle);

  // --- God mode: live world parameters (write straight into the params uniform) ---
  const godDefs: Omit<GodSpec, 'value'>[] = [
    // World
    { group: 'cat_world', labelKey: 'g_speed', idx: 3, min: 0.5, max: 10, step: 0.1 },
    { group: 'cat_world', labelKey: 'g_agility', idx: 4, min: 0.05, max: 1.2, step: 0.01 },
    { group: 'cat_world', labelKey: 'g_eatRange', idx: 6, min: 2, max: 24, step: 1 },
    { group: 'cat_world', labelKey: 'g_current', idx: 35, min: 0, max: 2, step: 0.05 },
    { group: 'cat_world', labelKey: 'g_phero', idx: 46, min: 0, max: 1024, step: 32 },
    // Food
    { group: 'cat_food', labelKey: 'g_food', idx: 15, min: 0, max: Math.max(8, n * 0.02), step: 1 },
    { group: 'cat_food', labelKey: 'g_foodEnergy', idx: 7, min: 2, max: 30, step: 0.5 },
    { group: 'cat_food', labelKey: 'g_patchiness', idx: 26, min: 0, max: 1, step: 0.05 },
    { group: 'cat_food', labelKey: 'g_bigFood', idx: 27, min: 1, max: 12, step: 0.5 },
    { group: 'cat_food', labelKey: 'g_bigFoodAmt', idx: 32, min: 0, max: 0.3, step: 0.005 },
    { group: 'cat_food', labelKey: 'g_carrion', idx: 38, min: 0, max: 1, step: 0.05 },
    // Evolution
    { group: 'cat_evolution', labelKey: 'g_mutRate', idx: 12, min: 0, max: 0.5, step: 0.01 },
    { group: 'cat_evolution', labelKey: 'g_mutSize', idx: 13, min: 0, max: 1, step: 0.02 },
    { group: 'cat_evolution', labelKey: 'g_reproAt', idx: 10, min: 30, max: 200, step: 1 },
    { group: 'cat_evolution', labelKey: 'g_offspring', idx: 14, min: 0.2, max: 0.8, step: 0.05 },
    { group: 'cat_evolution', labelKey: 'g_offspringSpread', idx: 33, min: 0, max: 20, step: 1 },
    { group: 'cat_evolution', labelKey: 'g_speciation', idx: 30, min: 0, max: 0.03, step: 0.001 },
    { group: 'cat_evolution', labelKey: 'g_sexual', idx: 45, min: 0, max: 1, step: 0.05 },
    { group: 'cat_evolution', labelKey: 'g_mate', idx: 47, min: 0, max: 4, step: 0.25 },
    // Body / metabolism
    { group: 'cat_body', labelKey: 'g_metabolism', idx: 8, min: 0, max: 0.6, step: 0.01 },
    { group: 'cat_body', labelKey: 'g_moveCost', idx: 9, min: 0, max: 0.3, step: 0.01 },
    { group: 'cat_body', labelKey: 'g_turnCost', idx: 31, min: 0, max: 0.2, step: 0.005 },
    { group: 'cat_body', labelKey: 'g_glowCost', idx: 34, min: 0, max: 0.2, step: 0.005 },
    { group: 'cat_body', labelKey: 'g_thermal', idx: 37, min: 0, max: 0.2, step: 0.005 },
    // Predation
    { group: 'cat_predation', labelKey: 'g_predation', idx: 24, min: 0, max: 1, step: 0.05 },
    { group: 'cat_predation', labelKey: 'g_predMargin', idx: 25, min: 1, max: 2.5, step: 0.05 },
    { group: 'cat_predation', labelKey: 'g_attackCost', idx: 36, min: 0, max: 0.2, step: 0.005 },
    { group: 'cat_predation', labelKey: 'g_toxin', idx: 39, min: 0, max: 50, step: 1 },
    // Cycle
    { group: 'cat_cycle', labelKey: 'g_dayNight', idx: 28, min: 0, max: 0.85, step: 0.05 },
    { group: 'cat_cycle', labelKey: 'g_dayLength', idx: 29, min: 400, max: 4000, step: 100 },
  ];
  // Restore shared god params into the uniform BEFORE warmup and before building
  // the sliders, so the warmed-up ocean and the slider positions both reflect the
  // link. Only known god indices are honoured, each value clamped to its slider
  // range (defends against a malformed/hostile hash).
  if (share?.params) {
    for (const { idx, value } of share.params) {
      const def = godDefs.find((d) => d.idx === idx);
      if (def) pf[idx] = Math.min(def.max, Math.max(def.min, value));
    }
    device.queue.writeBuffer(paramsBuf, 0, pbuf);
  }
  const godSpecs: GodSpec[] = godDefs.map((d) => ({ ...d, value: pf[d.idx]! }));
  const godPanel = buildGodPanel(godSpecs, (idx, value) => {
    pf[idx] = value;
    device.queue.writeBuffer(paramsBuf, 0, pbuf); // apply immediately (even if paused)
  });
  document.body.appendChild(godPanel.panel);
  ui.addTool(godPanel.toggle);

  // --- Mechanism toggles + scenario presets + random-world dice -------------
  // All drive the same god sliders, so changes are captured by sharing and the
  // slider thumbs stay in sync. applyParams writes the uniform once per action.
  const clampTo = (idx: number, v: number): number => {
    const d = godDefs.find((g) => g.idx === idx);
    return d ? Math.min(d.max, Math.max(d.min, v)) : v;
  };
  function applyParams(entries: [number, number][]): void {
    for (const [idx, value] of entries) {
      pf[idx] = value;
      godPanel.setValue(idx, value);
    }
    device.queue.writeBuffer(paramsBuf, 0, pbuf);
    refreshToggles();
  }

  const mkChip = (onClick: () => void): HTMLButtonElement => {
    const b = document.createElement('button');
    b.className = 'pg-chip';
    b.style.margin = '0 5px 5px 0';
    b.onclick = onClick;
    return b;
  };
  const mkSectionLabel = (key: string): HTMLElement => {
    const el = document.createElement('div');
    el.className = 'pg-eyebrow';
    el.style.cssText = 'margin:4px 0 8px;';
    el.dataset.i18n = key;
    el.textContent = t(key);
    return el;
  };

  // On-values for each toggleable mechanism (off = 0).
  const toggleDefs: { key: string; idx: number; on: number }[] = [
    { key: 'tog_predation', idx: 24, on: 0.5 },
    { key: 'tog_daynight', idx: 28, on: 0.45 },
    { key: 'tog_speciation', idx: 30, on: 0.004 },
    { key: 'tog_bigfood', idx: 32, on: 0.0625 },
    { key: 'tog_current', idx: 35, on: 0.4 },
    { key: 'tog_carrion', idx: 38, on: 0.3 },
  ];
  const toggleBtns: { btn: HTMLButtonElement; def: (typeof toggleDefs)[number] }[] = [];
  function refreshToggles(): void {
    for (const { btn, def } of toggleBtns) {
      const on = pf[def.idx]! > 0;
      btn.classList.toggle('is-on', on);
      const label = btn.querySelector('span:last-child');
      if (label) label.textContent = t(def.key);
    }
  }
  const togWrap = document.createElement('div');
  togWrap.style.cssText = 'margin-bottom:10px;display:flex;flex-wrap:wrap;gap:5px;';
  const togHead = mkSectionLabel('mechanisms');
  togHead.style.width = '100%';
  togWrap.append(togHead);
  for (const def of toggleDefs) {
    const btn = document.createElement('button');
    btn.className = 'pg-switch';
    btn.innerHTML = '<span class="pg-knob"></span><span></span>';
    btn.onclick = () => applyParams([[def.idx, pf[def.idx]! > 0 ? 0 : def.on]]);
    toggleBtns.push({ btn, def });
    togWrap.append(btn);
  }

  // Scenario presets: each overrides a few sliders for a distinct ecosystem.
  const bigFood = Math.max(8, Math.round(n * 0.012));
  const presetDefs: { key: string; set: Record<number, number> }[] = [
    { key: 'pre_eden', set: { 15: bigFood, 24: 0, 8: 0.08, 12: 0.06, 28: 0.15 } },
    { key: 'pre_famine', set: { 15: Math.max(2, Math.round(n * 0.0015)), 8: 0.38, 24: 0.5 } },
    { key: 'pre_carnage', set: { 24: 1, 25: 1.05, 15: Math.max(8, Math.round(n * 0.01)) } },
    { key: 'pre_soup', set: { 12: 0.4, 13: 0.8, 30: 0.02 } },
    { key: 'pre_night', set: { 28: 0.85, 29: 3500 } },
    { key: 'pre_titans', set: { 24: 0.8, 25: 1.4, 27: 12, 32: 0.12 } },
  ];
  const presetWrap = document.createElement('div');
  presetWrap.append(mkSectionLabel('scenarios'));
  const presetBtns: { btn: HTMLButtonElement; key: string }[] = [];
  for (const p of presetDefs) {
    const btn = mkChip(() =>
      applyParams(Object.entries(p.set).map(([k, v]) => [Number(k), clampTo(Number(k), v)])),
    );
    presetBtns.push({ btn, key: p.key });
    presetWrap.append(btn);
  }
  // Random world: roll every slider, biased away from instant extinction, then
  // reflect it in the address bar so the rolled ocean is shareable.
  const diceBtn = mkChip(() => {
    const entries: [number, number][] = godDefs.map((d) => {
      let lo = d.min;
      let hi = d.max;
      const span = d.max - d.min;
      if (d.idx === 15) lo = d.min + span * 0.4; // keep some food
      if (d.idx === 8 || d.idx === 9 || d.idx === 31) hi = d.min + span * 0.45; // tame costs
      if (d.idx === 24) hi = d.min + span * 0.7; // predation not maxed
      const raw = lo + Math.random() * (hi - lo);
      const snapped = Math.round(raw / d.step) * d.step;
      return [d.idx, Math.min(d.max, Math.max(d.min, snapped))];
    });
    applyParams(entries);
    applyHash({ seed, n, params: godPanel.getValues() });
  });
  presetWrap.append(diceBtn);

  godPanel.extras.append(togWrap, presetWrap);
  refreshToggles();
  function relabelGodExtras(): void {
    refreshToggles();
    for (const { btn, key } of presetBtns) btn.textContent = t(key);
    diceBtn.innerHTML = icon('shuffle', 14) + `<span>${t('pre_dice')}</span>`;
    for (const el of godPanel.extras.querySelectorAll<HTMLElement>('[data-i18n]')) {
      el.textContent = t(el.dataset.i18n!);
    }
  }
  relabelGodExtras();
  onLang(relabelGodExtras);

  // --- Share: copy a reproducible-ocean URL (seed + N + live god params) ---
  let shareFeedbackTimer = 0;
  const shareBtn = mkBtn('', () => void onShare());
  function relabelShare(): void {
    setBtnIcon(shareBtn, 'link', t('share'));
  }
  async function onShare(): Promise<void> {
    const state = { seed, n, params: godPanel.getValues() };
    applyHash(state); // reflect the current ocean in the address bar
    const ok = await copyToClipboard(buildShareUrl(state));
    window.clearTimeout(shareFeedbackTimer);
    setBtnIcon(shareBtn, ok ? 'check' : 'link', ok ? t('copied') : t('share'));
    shareFeedbackTimer = window.setTimeout(relabelShare, 1500);
  }
  relabelShare();
  onLang(relabelShare);
  ui.addTool(shareBtn);

  // --- Restart: reseed the ocean in place (same seed + current params, no reload) ---
  const resetBtn = mkBtn('', () => resetOcean());
  function relabelReset(): void {
    setBtnIcon(resetBtn, 'restart', t('restart'));
  }
  relabelReset();
  onLang(relabelReset);
  ui.addTool(resetBtn);

  // --- Help / pedagogical panel ---
  const help = document.createElement('div');
  help.style.cssText =
    'position:fixed;inset:0;display:none;place-items:center;z-index:1000;background:rgba(2,4,10,0.7);' +
    '-webkit-backdrop-filter:blur(3px);backdrop-filter:blur(3px);';
  const helpCard = document.createElement('div');
  helpCard.className = 'pg-panel';
  helpCard.style.cssText =
    'max-width:540px;margin:16px;padding:26px 28px;font:14px/1.7 var(--font-ui);position:relative;';
  const helpTitle = document.createElement('div');
  helpTitle.style.cssText =
    'font-size:19px;font-weight:600;letter-spacing:.02em;color:var(--glow-cyan);margin-bottom:14px;';
  const helpBody = document.createElement('div');
  helpBody.style.color = 'var(--ink-dim)';
  const helpClose = mkIconBtn('close', t('close'), () => (help.style.display = 'none'));
  helpClose.style.cssText += 'position:absolute;top:14px;right:14px;';
  helpCard.append(helpClose, helpTitle, helpBody);
  help.append(helpCard);
  help.addEventListener('click', (e) => {
    if (e.target === help) help.style.display = 'none';
  });
  document.body.appendChild(help);

  const helpToggle = mkBtn('', () => (help.style.display = 'grid'));

  function relabelHelp(): void {
    helpTitle.textContent = t('helpTitle');
    helpBody.innerHTML = t('helpBody');
    helpToggle.textContent = '?  ' + t('help');
    helpClose.title = t('close');
  }
  relabelHelp();
  onLang(relabelHelp);

  // Show the explainer once on first visit.
  if (!localStorage.getItem('pelagia-seen-help')) {
    help.style.display = 'grid';
    localStorage.setItem('pelagia-seen-help', '1');
  }
  // --- Observatory: full-screen data view (world history + lineages + watch-list) ---
  const MAX_WORLD_SAMPLES = 240; // ring length for every time series
  const MAX_TRACKED_LINEAGES = 80;
  interface LinTrack {
    hue: number;
    samples: number[];
    lastTick: number;
    count: number;
    trend: number;
    descKey: string;
    fast: boolean;
    seek: number;
    forage: number;
    cruise: number;
    aggression: number;
    neurons: number;
  }
  const worldSeries: WorldSample[] = [];
  const lineageHist = new Map<number, LinTrack>();
  const watched = new Map<number, Watched>();
  const parentMap = new Map<number, number>(); // new lineage id -> parent lineage id
  let watchPending = false;

  const observatory = buildObservatory((id) => removeWatch(id));
  document.body.appendChild(observatory.panel);
  ui.addTool(observatory.toggle);
  const history = buildEvolutionHistory();
  document.body.appendChild(history.panel);
  ui.addTool(history.toggle);

  // Highlight: dim everything but the selected creature's lineage (follow a clade).
  const highlightBtn = mkBtn('', () => {
    highlightOn = !highlightOn;
    applyHighlight();
    relabelHighlight();
  });
  function relabelHighlight(): void {
    setBtnIcon(highlightBtn, 'focus', t('highlight'));
    highlightBtn.classList.toggle('is-active', highlightOn);
  }
  relabelHighlight();
  onLang(relabelHighlight);
  ui.addTool(highlightBtn);

  // "Show fields": reveal the thermal biomes as a faint background tint.
  const fieldsBtn = mkBtn('', () => {
    fieldTint = fieldTint > 0 ? 0 : 0.14;
    relabelFields();
  });
  function relabelFields(): void {
    setBtnIcon(fieldsBtn, 'thermometer', t('showFields'));
    fieldsBtn.classList.toggle('is-active', fieldTint > 0);
  }
  relabelFields();
  onLang(relabelFields);
  ui.addTool(fieldsBtn);

  // "Follow": lock the camera onto the selected creature to watch its life up close.
  const followBtn = mkBtn('', () => {
    lockOn = !lockOn;
    relabelFollow();
  });
  function relabelFollow(): void {
    setBtnIcon(followBtn, 'crosshair', t('followCam'));
    followBtn.classList.toggle('is-active', lockOn);
  }
  relabelFollow();
  onLang(relabelFollow);
  ui.addTool(followBtn);

  // "Show currents": animated streaks revealing the flow field.
  const currentBtn = mkBtn('', () => {
    currentViz = !currentViz;
    relabelCurrent();
  });
  function relabelCurrent(): void {
    setBtnIcon(currentBtn, 'waves', t('showCurrents'));
    currentBtn.classList.toggle('is-active', currentViz);
  }
  relabelCurrent();
  onLang(relabelCurrent);
  ui.addTool(currentBtn);

  // "Show pheromones": reveal the trails creatures lay down as glowing green paths.
  const pheroBtn = mkBtn('', () => {
    pheroViz = !pheroViz;
    relabelPhero();
  });
  function relabelPhero(): void {
    setBtnIcon(pheroBtn, 'route', t('showPheromones'));
    pheroBtn.classList.toggle('is-active', pheroViz);
  }
  relabelPhero();
  onLang(relabelPhero);
  ui.addTool(pheroBtn);

  function addWatch(id: number, lineage: number): void {
    if (id < 0 || watched.has(id) || watched.size >= MAX_WATCH) return;
    watched.set(id, { id, lineage, hue: floatFromU32(pcgHash(lineage)), history: [] });
    pushObservatory();
  }
  function removeWatch(id: number): void {
    watched.delete(id);
    pushObservatory();
  }
  function buildObservatoryData(): ObservatoryData {
    const lineages: LineageHistory[] = [];
    for (const [lin, tr] of lineageHist) {
      lineages.push({
        lineage: lin,
        hue: tr.hue,
        count: tr.count,
        trend: tr.trend,
        descKey: tr.descKey,
        fast: tr.fast,
        seek: tr.seek,
        forage: tr.forage,
        cruise: tr.cruise,
        aggression: tr.aggression,
        neurons: tr.neurons,
        samples: tr.samples,
      });
    }
    return { world: worldSeries, lineages, watched: [...watched.values()], parents: parentMap };
  }
  function pushObservatory(): void {
    const obsData = buildObservatoryData();
    observatory.update(obsData);
    history.update(obsData);
  }

  // Sample the tracked creatures through the inspect pass (one tiny dispatch each;
  // capped at MAX_WATCH). Records energy/speed/decision history per individual.
  async function sampleWatched(): Promise<void> {
    if (watchPending) return;
    const ids = [...watched.keys()];
    if (ids.length === 0) return;
    watchPending = true;
    try {
      for (let k = 0; k < ids.length; k++) {
        pu[22] = ids[k]!;
        device.queue.writeBuffer(paramsBuf, 0, pbuf);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(inspectPipeline);
        pass.setBindGroup(0, inspectBind);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(
          inspectBuf,
          0,
          watchReadback,
          k * INSPECT_FLOATS * 4,
          INSPECT_FLOATS * 4,
        );
        device.queue.submit([enc.finish()]);
      }
      setSelectedUniform(); // restore the live selection index for the brain view
      device.queue.writeBuffer(paramsBuf, 0, pbuf);
      await watchReadback.mapAsync(GPUMapMode.READ);
      const d = new Float32Array(watchReadback.getMappedRange().slice(0));
      watchReadback.unmap();
      ids.forEach((id, k) => {
        const w = watched.get(id);
        if (!w) return;
        const o = k * INSPECT_FLOATS;
        const sameLineage = Math.round(d[o + 36]!) === w.lineage;
        const sample: WatchSample = {
          tick: frame,
          energy: d[o + 34]!,
          speed: d[o + 33]!,
          turn: d[o + 27]!,
          thrust: (d[o + 28]! + 1) / 2,
          alive: d[o + 37]! >= 0.5 && sameLineage,
        };
        w.history.push(sample);
        if (w.history.length > MAX_WORLD_SAMPLES) w.history.shift();
      });
    } finally {
      watchPending = false;
    }
  }

  // Maintain per-clade population curves on ONE shared time axis (every tracked
  // clade gets exactly one sample per call, 0 when absent), so they can be stacked
  // into a Muller plot. Track newly-seen clades; bound the map (evict long-extinct).
  function updateLineageHistories(
    counts: Map<number, number>,
    poolMeta: Map<number, LineageTraits>,
  ): void {
    // 1. Start tracking newly-seen, sizeable/characterised clades.
    for (const [lin, count] of counts) {
      if (lineageHist.has(lin)) continue;
      const meta = poolMeta.get(lin);
      if (!meta && count < 2) continue;
      lineageHist.set(lin, {
        hue: floatFromU32(pcgHash(lin)),
        samples: [],
        lastTick: frame,
        count: 0,
        trend: 0,
        descKey: meta?.descKey ?? 'desc_erratic',
        fast: meta?.fast ?? false,
        seek: meta?.seek ?? 0,
        forage: meta?.forage ?? 0,
        cruise: meta?.cruise ?? 0,
        aggression: meta?.aggression ?? 0,
        neurons: meta?.neurons ?? 0,
      });
    }
    // 2. Append one aligned sample to every tracked clade + refresh its traits.
    for (const [lin, tr] of lineageHist) {
      const c = counts.get(lin) ?? 0;
      const meta = poolMeta.get(lin);
      if (meta) {
        tr.descKey = meta.descKey;
        tr.fast = meta.fast;
        tr.seek = meta.seek;
        tr.forage = meta.forage;
        tr.cruise = meta.cruise;
        tr.aggression = meta.aggression;
        tr.neurons = meta.neurons;
      }
      tr.trend = c - tr.count;
      tr.count = c;
      if (c > 0) tr.lastTick = frame;
      tr.samples.push(c);
      if (tr.samples.length > MAX_WORLD_SAMPLES) tr.samples.shift();
    }
    // 3. Bound the map (evict long-extinct, then smallest).
    if (lineageHist.size > MAX_TRACKED_LINEAGES) {
      const entries = [...lineageHist.entries()].sort((a, b) => {
        const da = a[1].count === 0 ? 0 : 1;
        const db = b[1].count === 0 ? 0 : 1;
        if (da !== db) return da - db; // dead clades first
        if (da === 0) return a[1].lastTick - b[1].lastTick; // oldest-dead first
        return a[1].count - b[1].count; // then smallest alive
      });
      for (const [lin] of entries) {
        if (lineageHist.size <= MAX_TRACKED_LINEAGES) break;
        lineageHist.delete(lin);
      }
    }
  }

  let analysisPending = false;
  let lastAnalysisT = 0;
  const prevCounts = new Map<number, number>();

  // Periodic world sample: reads bio + food back, drives the lineage panel AND the
  // observatory (world time series, lineage histories, tracked creatures).
  async function sampleWorld(): Promise<void> {
    if (analysisPending) return;
    analysisPending = true;
    try {
      const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
      const fRead = device.createBuffer({ size: f * 8, usage: GPUBufferUsage.MAP_READ | CD });
      // Speciation counter + parent-lineage pointers, for the family tree.
      const spRead = device.createBuffer({
        size: (1 + n) * 4,
        usage: GPUBufferUsage.MAP_READ | CD,
      });
      let enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
      enc.copyBufferToBuffer(foodBuf, 0, fRead, 0, f * 8);
      enc.copyBufferToBuffer(gridDataBuf, speciesCountOffset, spRead, 0, (1 + n) * 4);
      device.queue.submit([enc.finish()]);
      await Promise.all([
        bRead.mapAsync(GPUMapMode.READ),
        fRead.mapAsync(GPUMapMode.READ),
        spRead.mapAsync(GPUMapMode.READ),
      ]);
      const bi = new Float32Array(bRead.getMappedRange().slice(0));
      const fo = new Float32Array(fRead.getMappedRange().slice(0));
      const sp = new Uint32Array(spRead.getMappedRange().slice(0));
      bRead.unmap();
      bRead.destroy();
      let foodAlive = 0;
      for (let j = 0; j < f; j++) if (fo[j * 2]! >= 0) foodAlive++;
      fRead.unmap();
      spRead.unmap();
      spRead.destroy();
      // Rebuild the parent map: new lineage (n + k) descends from parent sp[1 + k].
      parentMap.clear();
      const newCount = Math.min(sp[0]!, n);
      for (let k = 0; k < newCount; k++) parentMap.set(n + k, sp[1 + k]!);
      fRead.destroy();

      const counts = new Map<number, number>();
      const repSlot = new Map<number, number>();
      let aliveCount = 0;
      let sumE = 0;
      let minE = Infinity;
      let maxE = -Infinity;
      for (let i = 0; i < n; i++) {
        if (bi[i * 4 + 2]! < 0.5) continue;
        aliveCount++;
        const e = bi[i * 4]!;
        sumE += e;
        if (e < minE) minE = e;
        if (e > maxE) maxE = e;
        const lin = Math.round(bi[i * 4 + 3]!);
        counts.set(lin, (counts.get(lin) ?? 0) + 1);
        if (!repSlot.has(lin)) repSlot.set(lin, i);
      }
      const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
      const pool = ranked.slice(0, 16); // characterise a pool, then pick groups

      const rows: LineageRow[] = [];
      const strategy: Record<string, number> = {};
      const poolMeta = new Map<number, LineageTraits>();
      let meanSize = 1;
      if (pool.length > 0) {
        const gRead = device.createBuffer({
          size: pool.length * GENOME_SIZE * 4,
          usage: GPUBufferUsage.MAP_READ | CD,
        });
        enc = device.createCommandEncoder();
        pool.forEach(([lin], k) => {
          enc.copyBufferToBuffer(
            weightsBuf,
            repSlot.get(lin)! * GENOME_SIZE * 4,
            gRead,
            k * GENOME_SIZE * 4,
            GENOME_SIZE * 4,
          );
        });
        device.queue.submit([enc.finish()]);
        await gRead.mapAsync(GPUMapMode.READ);
        const gAll = new Float32Array(gRead.getMappedRange().slice(0));
        gRead.unmap();
        gRead.destroy();

        const traits = pool.map((_e, k) =>
          characterizeGenome(gAll.subarray(k * GENOME_SIZE, (k + 1) * GENOME_SIZE)),
        );
        const dominantCount = Math.min(5, pool.length);
        // Behavioural centroid of the dominant clades; the "distinct" section
        // surfaces the non-dominant survivors whose policy differs MOST from it
        // (real behavioural variety, not just the smallest clades — which, after
        // convergence, tend to be ordinary too).
        const axes = (tr: LineageTraits): number[] => [tr.seek, tr.forage, tr.cruise, tr.turnBias];
        const centroid = [0, 0, 0, 0];
        for (let k = 0; k < dominantCount; k++) {
          const a = axes(traits[k]!);
          for (let d = 0; d < 4; d++) centroid[d]! += a[d]! / dominantCount;
        }
        const dist = (a: number[]): number =>
          Math.hypot(
            a[0]! - centroid[0]!,
            a[1]! - centroid[1]!,
            a[2]! - centroid[2]!,
            a[3]! - centroid[3]!,
          );
        const distinctIdx: number[] = [];
        for (let k = dominantCount; k < pool.length; k++) distinctIdx.push(k);
        distinctIdx.sort((a, b) => dist(axes(traits[b]!)) - dist(axes(traits[a]!)));

        const pushRow = (k: number, group: 'dominant' | 'distinct'): void => {
          const [lin, count] = pool[k]!;
          const ch = traits[k]!;
          rows.push({
            lineage: lin,
            hue: floatFromU32(pcgHash(lin)),
            count,
            trend: count - (prevCounts.get(lin) ?? count),
            group,
            descKey: ch.descKey,
            fast: ch.fast,
            seek: ch.seek,
            bigSeek: ch.bigSeek,
            forage: ch.forage,
            cruise: ch.cruise,
            aggression: ch.aggression,
            neurons: ch.neurons,
          });
        };
        for (let k = 0; k < dominantCount; k++) pushRow(k, 'dominant');
        for (const k of distinctIdx.slice(0, 3)) pushRow(k, 'distinct');

        prevCounts.clear();
        for (const [lin, count] of counts) prevCounts.set(lin, count);

        // Strategy mix snapshot + per-lineage trait metadata for the observatory,
        // plus a population-weighted mean body size for the narrative.
        let szSum = 0;
        let cSum = 0;
        pool.forEach(([lin, count], k) => {
          const tr = traits[k]!;
          strategy[tr.descKey] = (strategy[tr.descKey] ?? 0) + count;
          poolMeta.set(lin, tr);
          szSum += tr.size * count;
          cSum += count;
        });
        if (cSum > 0) meanSize = szSum / cSum;
      }
      lineagePanel.update(rows);

      // --- Observatory time series ---
      updateLineageHistories(counts, poolMeta);
      worldSeries.push({
        tick: frame,
        alive: aliveCount,
        foodAlive,
        lineages: counts.size,
        energyAvg: aliveCount ? sumE / aliveCount : 0,
        energyMin: aliveCount ? minE : 0,
        energyMax: aliveCount ? maxE : 0,
        predKills: lastPredKills,
        predGain: pf[24]!,
        meanSize,
        daylight:
          pf[28]! > 0 ? 0.5 + 0.5 * Math.sin((2 * Math.PI * frame) / Math.max(1, pf[29]!)) : -1,
        strategy,
      });
      if (worldSeries.length > MAX_WORLD_SAMPLES) worldSeries.shift();
      await sampleWatched();
      pushObservatory();
    } finally {
      analysisPending = false;
    }
  }

  // Occasional non-blocking readback of the alive count (= n - freeCount) and the
  // per-tick predation kill count (gridData[2*numCells .. +3] = free, budget, pred).
  let alive = n;
  let lastPredKills = 0;
  let countersPending = false;
  const countersReadback = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.MAP_READ | CD,
  });
  function pollAlive(): void {
    if (countersPending) return;
    countersPending = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gridDataBuf, 2 * numCells * 4, countersReadback, 0, 12);
    device.queue.submit([enc.finish()]);
    void countersReadback.mapAsync(GPUMapMode.READ).then(() => {
      const c = new Uint32Array(countersReadback.getMappedRange());
      alive = n - c[0]!;
      lastPredKills = c[2]!;
      countersReadback.unmap();
      countersPending = false;
    });
  }

  // --- Chemotaxis validation hook (read GPU state back, compute on CPU) ---
  (globalThis as unknown as { __pelagia: unknown }).__pelagia = {
    n,
    f,
    seed,
    get tick() {
      return frame;
    },
    readChemotaxis,
  };
  async function readChemotaxis(): Promise<unknown> {
    const sRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const bRead = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | CD });
    const fRead = device.createBuffer({ size: f * 8, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(stateBuf, 0, sRead, 0, n * 16);
    enc.copyBufferToBuffer(bioBuf, 0, bRead, 0, n * 16);
    enc.copyBufferToBuffer(foodBuf, 0, fRead, 0, f * 8);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      sRead.mapAsync(GPUMapMode.READ),
      bRead.mapAsync(GPUMapMode.READ),
      fRead.mapAsync(GPUMapMode.READ),
    ]);
    const st = new Float32Array(sRead.getMappedRange().slice(0));
    const bi = new Float32Array(bRead.getMappedRange().slice(0));
    const fo = new Float32Array(fRead.getMappedRange().slice(0));
    sRead.unmap();
    bRead.unmap();
    fRead.unmap();

    const fx = new Float32Array(f);
    const fy = new Float32Array(f);
    for (let j = 0; j < f; j++) {
      fx[j] = fo[j * 2]!;
      fy[j] = fo[j * 2 + 1]!;
    }
    const grid = new SpatialGrid(world, world, cellSize);
    grid.build(fx, fy, f);

    let count = 0;
    let sumCos = 0;
    let seeking = 0;
    let aliveCount = 0;
    let sumE = 0;
    let minE = Infinity;
    let maxE = -Infinity;
    for (let i = 0; i < n; i++) {
      if (bi[i * 4 + 2]! < 0.5) continue;
      aliveCount++;
      const e = bi[i * 4]!;
      sumE += e;
      if (e < minE) minE = e;
      if (e > maxE) maxE = e;
      const px = st[i * 4]!;
      const py = st[i * 4 + 1]!;
      const heading = st[i * 4 + 2]!;
      const fj = grid.findNearest(px, py, cellSize);
      if (fj < 0) continue;
      const dx = wrapDelta(fx[fj]! - px, world);
      const dy = wrapDelta(fy[fj]! - py, world);
      const c = Math.cos(Math.atan2(dy, dx) - heading);
      sumCos += c;
      if (c > 0.5) seeking++;
      count++;
    }
    let aliveFood = 0;
    for (let j = 0; j < f; j++) if (fx[j]! >= 0) aliveFood++;
    return {
      tick: frame,
      c0: {
        x: +st[0]!.toFixed(1),
        y: +st[1]!.toFixed(1),
        heading: +st[2]!.toFixed(2),
        energy: bi[0]!,
        lineage: bi[3]!,
      },
      alive: aliveCount,
      aliveFood,
      withFood: count,
      energy: {
        min: +minE.toFixed(1),
        avg: +(sumE / Math.max(1, aliveCount)).toFixed(1),
        max: +maxE.toFixed(1),
      },
      chemotaxis: +(sumCos / Math.max(1, count)).toFixed(3),
      fracSeeking: +(seeking / Math.max(1, count)).toFixed(3),
    };
  }

  let frame = 0;
  let frames = 0;
  let stepAcc = 0;
  let lastFpsT = performance.now();

  // Restart the ocean in place: reseed the buffers from the current seed, empty
  // the free-slot stack, rewind the frame counter (so the shader RNG replays),
  // wipe trails and drop any selection. Same seed + current god params = a
  // reproducible re-run; revives everyone after an extinction without a reload.
  function resetOcean(): void {
    fillInitialData();
    device.queue.writeBuffer(stateBuf, 0, stateData);
    device.queue.writeBuffer(bioBuf, 0, bioData);
    device.queue.writeBuffer(weightsBuf, 0, weightsData);
    device.queue.writeBuffer(foodBuf, 0, foodData);
    device.queue.writeBuffer(gridDataBuf, 2 * numCells * 4, new Uint32Array([0])); // freeCount = 0
    device.queue.writeBuffer(gridDataBuf, speciesCountOffset, new Uint32Array([0])); // speciation counter = 0
    frame = 0;
    stepAcc = 0;
    alive = n;
    prevCounts.clear();
    lineagePanel.update([]);
    deselect();
    const enc = device.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: accumView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      .end();
    device.queue.submit([enc.finish()]);
  }

  // Headless warm-up: advance the simulation before the first rendered frame so
  // the ocean lands in a chosen state (past the initial die-off, mid-evolution).
  const warmup = opts.warmup ?? 0;
  for (let w = 0; w < warmup; w++) {
    writeParams(frame);
    const enc = device.createCommandEncoder();
    recordSim(enc);
    device.queue.submit([enc.finish()]);
    frame++;
  }
  opts.onReady?.();

  function tick(): void {
    // Keep the drawing buffer in sync with the viewport (handles a late initial
    // layout where innerWidth was 0, and any window resize).
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const wantW = Math.max(1, Math.floor(window.innerWidth * dpr));
    const wantH = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (canvas.width !== wantW || canvas.height !== wantH) resize();

    // Simulation steps. Speed may be < 1 (slow motion): accumulate fractional
    // steps so e.g. 0.5x runs one tick every other frame. Each step is its own
    // submit so the frame counter (used for RNG keying) varies per tick.
    let steps = 0;
    if (!ui.paused) {
      stepAcc += ui.speed;
      steps = Math.floor(stepAcc);
      stepAcc -= steps;
    }
    for (let k = 0; k < steps; k++) {
      writeParams(frame);
      const enc = device.createCommandEncoder();
      recordSim(enc);
      device.queue.submit([enc.finish()]);
      frame++;
    }

    // Day/night: dim the scene on the same sine that swings the food influx, so
    // the cycle is visible (the ocean brightens by day, darkens at night).
    const dayPhase = Math.sin((2 * Math.PI * frame) / Math.max(1, pf[29]!));
    renderData[5] = 1.4 * (1 + 0.3 * pf[28]! * dayPhase);
    renderData[7] = Math.floor(f * pf[32]!); // big-food count tracks the live slider
    renderData[10] = fieldTint; // thermal-field background tint (0 = off)
    renderData[12] = frame; // for the present pass's tempAt drift
    renderData[13] = pf[35]!; // current strength (for the flow-streak reveal)
    renderData[14] = currentViz ? 1 : 0; // current-streak overlay on/off
    renderData[17] = pheroViz ? 1 : 0; // pheromone-field overlay on/off
    device.queue.writeBuffer(renderUbo, 0, renderData);

    // Render every frame (so trails keep fading even while paused).
    const encoder = device.createCommandEncoder();
    const accumPass = encoder.beginRenderPass({
      colorAttachments: [{ view: accumView, loadOp: 'load', storeOp: 'store' }],
    });
    accumPass.setPipeline(fadePipeline);
    accumPass.draw(3);
    accumPass.setPipeline(foodPipeline);
    accumPass.setBindGroup(0, foodBind);
    accumPass.draw(6, f);
    accumPass.setPipeline(creaturePipeline);
    accumPass.setBindGroup(0, creatureBind);
    accumPass.draw(6, n);
    accumPass.end();

    const view = context!.getCurrentTexture().createView();
    const present = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    present.setPipeline(presentPipeline);
    present.setBindGroup(0, presentBind);
    present.draw(3);
    present.end();
    device.queue.submit([encoder.finish()]);

    // Brain inspector: recompute the selected creature's brain and read it back.
    if (selectedIndex >= 0 && !inspectPending) {
      setSelectedUniform();
      writeParams(frame); // includes pu[22] = selectedIndex
      const inspectFrame = frame; // tag the sample so the decision tape advances per tick
      const ienc = device.createCommandEncoder();
      const ipass = ienc.beginComputePass();
      ipass.setPipeline(inspectPipeline);
      ipass.setBindGroup(0, inspectBind);
      ipass.dispatchWorkgroups(1);
      ipass.end();
      ienc.copyBufferToBuffer(inspectBuf, 0, inspectReadback, 0, INSPECT_FLOATS * 4);
      device.queue.submit([ienc.finish()]);
      inspectPending = true;
      void inspectReadback.mapAsync(GPUMapMode.READ).then(() => {
        const d = new Float32Array(inspectReadback.getMappedRange().slice(0));
        inspectReadback.unmap();
        inspectPending = false;
        if (selectedIndex < 0) return;
        const alive = d[37]! >= 0.5;
        const sameLineage = Math.round(d[36]!) === selectedLineage;
        if (alive && sameLineage) {
          lastGoodSample = d.slice(); // remember the last living frame
          brainView.update(d, inspectFrame);
          if (lockOn) {
            cam.cx = d[30]!; // creature world x
            cam.cy = d[31]!; // creature world y
            clampCam();
            updateView();
          }
          positionRing(d);
        } else {
          // The selected creature died. Either its slot is now empty (same lineage,
          // not alive) or a newborn has already taken its slot (different lineage).
          // Either way, freeze the panel on its final state shown as deceased —
          // never let a selected creature silently vanish — then stop following.
          const finalSample = sameLineage ? d : lastGoodSample;
          ring.style.display = 'none';
          if (finalSample) {
            finalSample[37] = 0; // mark deceased
            brainView.update(finalSample, inspectFrame);
            selectedIndex = -1;
            setSelectedUniform();
          } else {
            // Died before we ever captured a living frame — nothing to show.
            deselect();
          }
        }
      });
    }

    frames++;
    const now = performance.now();
    if (now - lastFpsT >= 500) {
      const fps = (frames * 1000) / (now - lastFpsT);
      frames = 0;
      lastFpsT = now;
      pollAlive();
      ui.update(alive, fps, frame);
      if (now - lastAnalysisT > 1500) {
        lastAnalysisT = now;
        void sampleWorld();
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
