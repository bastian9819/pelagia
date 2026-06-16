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
  let world = vec2<f32>(s.x, s.y) + q * R.sizeWorld;
  out.pos = vec4<f32>(world.x * R.view.x + R.view.z, world.y * R.view.y + R.view.w, 0.0, 1.0);
  out.uv = q;
  // Slightly brighter when younger/faster could come later; keep hue for now.
  out.color = hue2rgb(b.y) * R.brightness;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  let halo = smoothstep(1.0, 0.0, d); // soft outer glow
  let core = smoothstep(0.4, 0.0, d); // bright core
  let intensity = halo * halo * 0.5 + core * 1.4;
  return vec4<f32>(in.color * intensity, 1.0); // additive blend
}
