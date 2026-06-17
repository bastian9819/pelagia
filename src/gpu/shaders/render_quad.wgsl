// Creatures as soft glowing billboards (additive into an HDR accum texture).
// Reads state + bio directly; dead/free slots are pushed off-screen.

struct RParams {
  view: vec4<f32>, // sx, sy, ox, oy
  sizeWorld: f32,  // glow radius in world units
  brightness: f32,
  _p0: f32,
  _p1: f32,
};

@group(0) @binding(0) var<uniform> R: RParams;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bio: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>; // for the body-size gene

const SIZE_GENE: u32 = 122u;
const GENOME_SIZE: u32 = 123u;

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
  out.color = hue2rgb(b.y) * R.brightness;
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
