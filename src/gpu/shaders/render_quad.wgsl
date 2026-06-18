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

const WEIGHT_GENES: u32 = 142u;
const HIDDEN_SIZE: u32 = 10u;
const SIZE_GENE: u32 = 152u;
const GENOME_SIZE: u32 = 153u;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
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
  // Orient the billboard so local +y points along the heading (swim direction).
  let a = s.z - 1.5707963;
  let ca = cos(a);
  let sa = sin(a);
  let rl = vec2<f32>(q.x * ca - q.y * sa, q.x * sa + q.y * ca);
  // Scale the body by the evolved size gene so morphology is visible on screen.
  let bodySize = clamp(1.0 + 0.5 * weights[ii * GENOME_SIZE + SIZE_GENE], 0.6, 2.2);
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
  }
  // Highlight mode: dim every creature that isn't in the selected lineage, so a
  // clade stands out among thousands.
  var dim = 1.0;
  if (R.hl.y > 0.5 && abs(b.w - R.hl.x) > 0.5) { dim = 0.1; }
  out.color = col * R.brightness * dim;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let u = in.uv;
  // Tadpole body: narrower across (x), elongated along the heading (y).
  let body = smoothstep(1.0, 0.0, length(vec2<f32>(u.x * 1.7, u.y)));
  // Bright head toward the front (+y); a faint tail behind it.
  let head = smoothstep(0.55, 0.0, length(vec2<f32>(u.x * 1.6, u.y - 0.35)));
  let intensity = body * body * 0.45 + head * 1.5;
  return vec4<f32>(in.color * intensity, 1.0); // additive blend
}
