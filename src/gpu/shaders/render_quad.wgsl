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
  @location(2) fin: f32, // tail-filament length (cosmetic morphology)
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
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let u = in.uv;
  // Tadpole body: narrower across (x), elongated along the heading (y).
  let body = smoothstep(1.0, 0.0, length(vec2<f32>(u.x * 1.7, u.y)));
  // Bright head toward the front (+y); a faint tail behind it.
  let head = smoothstep(0.55, 0.0, length(vec2<f32>(u.x * 1.6, u.y - 0.35)));
  // Evolved tail filament: a thin glowing streak trailing behind, length/strength
  // set by the FIN gene — pure silhouette variety so creatures look distinct.
  let tail = smoothstep(0.14, 0.0, abs(u.x)) * smoothstep(-0.2, -1.0, u.y) * in.fin;
  let intensity = body * body * 0.45 + head * 1.5 + tail * 0.7;
  return vec4<f32>(in.color * intensity, 1.0); // additive blend
}
