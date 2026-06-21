// Creatures as soft glowing billboards (additive into an HDR accum texture).
// Reads state + bio directly; dead/free slots are pushed off-screen.

struct RParams {
  view: vec4<f32>, // sx, sy, ox, oy
  sizeWorld: f32,  // glow radius in world units
  brightness: f32,
  colorMode: f32,  // 0 lineage, 1 size, 2 neurons, 3 energy, 4 speed
  _p1: f32,
  hl: vec4<f32>,   // highlight: x = lineage id, y = on (>0.5), _, _
};

@group(0) @binding(0) var<uniform> R: RParams;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bio: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>; // genome: size + activation genes

const WEIGHT_GENES: u32 = 213u;
const HIDDEN_SIZE: u32 = 10u;
const SIZE_GENE: u32 = 223u;
const ELONG_GENE: u32 = 224u;
const FIN_GENE: u32 = 225u;
const GLOW_GENE: u32 = 226u;
const THERMAL_GENE: u32 = 227u;
const TOXIN_GENE: u32 = 228u;
const GENOME_SIZE: u32 = 229u;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) fin: f32,  // caudal-fin size (FIN gene)
  @location(3) seed: f32, // per-lineage seed (drives the body pattern)
};

fn hue2rgb(h: f32) -> vec3<f32> {
  let r = abs(h * 6.0 - 3.0) - 1.0;
  let g = 2.0 - abs(h * 6.0 - 2.0);
  let b = 2.0 - abs(h * 6.0 - 4.0);
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );
  var out: VSOut;
  let b = bio[ii];
  if (b.z < 0.5) {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.color = vec3<f32>(0.0);
    return out;
  }
  let q = quad[vi];
  let s = state[ii];
  // Evolved morphology genes (all phenotype, visible on screen).
  let g = ii * GENOME_SIZE;
  let bodySize = clamp(1.0 + 0.5 * weights[g + SIZE_GENE], 0.6, 2.2);
  let elong = clamp(1.0 + 0.6 * weights[g + ELONG_GENE], 0.5, 2.0);
  let glow = clamp(1.0 + 0.6 * weights[g + GLOW_GENE], 0.6, 2.0);
  out.fin = clamp(0.5 + 0.5 * weights[g + FIN_GENE], 0.0, 1.0);
  // Orient the billboard so local +y points along the heading (swim direction).
  // Stretch along the body axis by elongation (area-preserving): eels look long
  // and thin, blobs short and wide — morphology is legible at a glance.
  let a = s.z - 1.5707963;
  let ca = cos(a);
  let sa = sin(a);
  let qs = vec2<f32>(q.x / sqrt(elong), q.y * sqrt(elong));
  let rl = vec2<f32>(qs.x * ca - qs.y * sa, qs.x * sa + qs.y * ca);
  let world = vec2<f32>(s.x, s.y) + rl * R.sizeWorld * bodySize;
  out.pos = vec4<f32>(world.x * R.view.x + R.view.z, world.y * R.view.y + R.view.w, 0.0, 1.0);
  out.uv = q; // unrotated local coords; +y is the creature's front

  // Colour by lineage (default) or by an evolving trait, so the whole ocean
  // visibly shifts hue as a trait spreads. Trait ramp: blue (low) -> red (high).
  let mode = u32(R.colorMode + 0.5);
  var col = hue2rgb(b.y);
  if (mode == 1u) {
    col = hue2rgb((1.0 - (bodySize - 0.6) / 1.6) * 0.66);
  } else if (mode == 2u) {
    var on = 0.0;
    for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
      if (weights[ii * GENOME_SIZE + WEIGHT_GENES + h] >= 0.0) { on = on + 1.0; }
    }
    col = hue2rgb((1.0 - on / f32(HIDDEN_SIZE)) * 0.66);
  } else if (mode == 3u) {
    col = hue2rgb((1.0 - clamp(b.x / 100.0, 0.0, 1.0)) * 0.66);
  } else if (mode == 4u) {
    col = hue2rgb((1.0 - clamp(s.w / 4.0, 0.0, 1.0)) * 0.66);
  } else if (mode == 5u) {
    col = hue2rgb((1.0 - (elong - 0.5) / 1.5) * 0.66); // elongation
  } else if (mode == 6u) {
    col = hue2rgb((1.0 - (glow - 0.6) / 1.4) * 0.66); // bioluminescence
  } else if (mode == 7u) {
    let pref = clamp(weights[g + THERMAL_GENE], -1.0, 1.0);
    col = hue2rgb((1.0 - (pref + 1.0) * 0.5) * 0.66); // thermal preference: cold=blue, warm=red
  } else if (mode == 8u) {
    // Aposematism: toxic creatures glow warning yellow-green, harmless ones dim grey.
    let tox = clamp(weights[g + TOXIN_GENE], 0.0, 1.0);
    col = mix(vec3<f32>(0.18, 0.20, 0.22), vec3<f32>(0.7, 1.0, 0.1), tox);
  }
  // Highlight mode: dim every creature that isn't in the selected lineage, so a
  // clade stands out among thousands.
  var dim = 1.0;
  if (R.hl.y > 0.5 && abs(b.w - R.hl.x) > 0.5) { dim = 0.1; }
  // Brighter creatures literally glow more (evolved bioluminescence).
  out.color = col * R.brightness * dim * glow;
  out.seed = b.y; // lineage hue doubles as a stable per-family pattern seed
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  // Local coords: +y is the front (head), -y the back (tail). The billboard is
  // already oriented + stretched by elongation in the vertex stage.
  let x = in.uv.x;
  let t = in.uv.y;
  let fin = in.fin;

  // Per-lineage BODY PLAN: the lineage seed picks one of three archetypes, so
  // different clades read as different kinds of creature (and speciation spawns new
  // looks). The evolved genes (elongation, fin, glow) still shape each individual.
  let sel = fract(in.seed * 3.71 + 0.19);

  // Family-specific banding pattern (stronger than before, for more texture).
  let freq = 6.0 + fract(in.seed * 9.0) * 11.0;
  let bands = 0.74 + 0.26 * sin(t * freq + in.seed * 6.28318);

  var intensity = 0.0;

  if (sel < 0.36) {
    // ---------- LEAF / PETAL: a graceful glowing teardrop ----------
    let prof = clamp(smoothstep(-0.92, -0.15, t) * smoothstep(0.72, -0.05, t), 0.0, 1.0);
    let halfW = 0.4 * pow(prof, 0.62);
    let edge = halfW - abs(x);
    let body = smoothstep(0.0, 0.12, edge);
    let spine = smoothstep(0.12, 0.0, abs(x)) * body;
    let rim = smoothstep(0.05, 0.0, abs(edge)) * step(0.04, prof);
    let head = smoothstep(0.32, 0.0, length(vec2<f32>(x * 1.25, t - 0.4)));
    let tb = -t - 0.5;
    let fl = 0.28 + 0.4 * fin;
    let flare = smoothstep(0.0, 0.05, tb) * smoothstep(fl, 0.0, tb);
    let tailHalf = 0.03 + flare * (0.09 + 0.28 * fin);
    let caudal = smoothstep(0.045, 0.0, abs(x) - tailHalf) * step(0.0, tb);
    intensity = body * 0.46 * bands + spine * 0.5 + rim * 0.75 + head * 1.7 + caudal * 0.8;
  } else if (sel < 0.7) {
    // ---------- FISH: streamlined body + bright head + a clear FORKED tail ----------
    let prof = clamp(smoothstep(-0.55, -0.05, t) * smoothstep(0.85, -0.1, t), 0.0, 1.0);
    let halfW = 0.27 * pow(prof, 0.55);
    let edge = halfW - abs(x);
    let body = smoothstep(0.0, 0.11, edge);
    let spine = smoothstep(0.1, 0.0, abs(x)) * body;
    let rim = smoothstep(0.05, 0.0, abs(edge)) * step(0.04, prof);
    let head = smoothstep(0.28, 0.0, length(vec2<f32>(x * 1.3, t - 0.36)));
    // Forked caudal tail: a fan bright at its two outer lobes and dim in the central
    // notch — the iconic fish-tail read. Size set by the FIN gene.
    let tb = -t - 0.4;
    let fl = 0.32 + 0.42 * fin;
    let env = smoothstep(0.0, 0.05, tb) * smoothstep(fl, 0.0, tb);
    let fanHalf = env * (0.15 + 0.45 * fin);
    let lobe = smoothstep(0.05, 0.0, abs(x) - fanHalf) * step(0.0, tb);
    let notch = smoothstep(0.0, 0.13, abs(x)); // 0 in the centre, 1 toward the lobes
    let caudal = lobe * (0.3 + 0.7 * notch);
    intensity = body * 0.44 * bands + spine * 0.5 + rim * 0.7 + head * 1.6 + caudal * 1.0;
  } else {
    // ---------- JELLY / MEDUSA: a bright bell with soft trailing tentacles ----------
    let d = length(vec2<f32>(x * 1.05, (t - 0.2) * 0.9));
    let bell = smoothstep(0.55, 0.0, d);
    let bellRim = smoothstep(0.08, 0.0, abs(d - 0.5)) * step(t, 0.3);
    // a few softly-waving tentacle filaments trailing behind the bell (FIN = length)
    var tent = 0.0;
    let len = 0.7 + 0.5 * fin;
    for (var k = 0; k < 3; k = k + 1) {
      let off = (f32(k) - 1.0) * 0.18;
      let wob = 0.1 * sin(t * 7.0 + f32(k) * 2.1 + in.seed * 18.0);
      let along = smoothstep(0.2, 0.2 - len, t); // fade with distance behind the bell
      tent = tent + smoothstep(0.035, 0.0, abs(x - off - wob)) * along;
    }
    tent = tent * step(t, 0.24) * (0.5 + 0.5 * fin);
    intensity = bell * 1.55 * bands + bellRim * 0.8 + tent * 0.55;
  }

  return vec4<f32>(in.color * intensity, 1.0); // additive blend
}
