// Instanced creature renderer: one small heading-oriented triangle per
// creature, read straight from the GPU state buffer (zero-copy — no readback).
// Additive blending gives the bioluminescent glow where creatures overlap.

struct RParams {
  view: vec4<f32>, // sx, sy, ox, oy  (world -> NDC: ndc = world*scale + offset)
  pointSize: f32,
  brightness: f32,
  _p0: f32,
  _p1: f32,
};

@group(0) @binding(0) var<uniform> R: RParams;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>; // x, y, heading, speed
@group(0) @binding(2) var<storage, read> hue: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
};

fn hue2rgb(h: f32) -> vec3<f32> {
  let r = abs(h * 6.0 - 3.0) - 1.0;
  let g = 2.0 - abs(h * 6.0 - 2.0);
  let b = 2.0 - abs(h * 6.0 - 4.0);
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var tri = array<vec2<f32>, 3>(vec2<f32>(0.0, 1.4), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0));
  let s = state[ii];
  let local = tri[vi] * R.pointSize;

  // Triangle points +y locally; rotate so +y aligns with the heading.
  let a = s.z - 1.5707963;
  let ca = cos(a);
  let sa = sin(a);
  let rl = vec2<f32>(local.x * ca - local.y * sa, local.x * sa + local.y * ca);

  let world = vec2<f32>(s.x, s.y) + rl;
  let ndc = vec2<f32>(world.x * R.view.x + R.view.z, world.y * R.view.y + R.view.w);

  var out: VSOut;
  out.pos = vec4<f32>(ndc, 0.0, 1.0);
  out.color = hue2rgb(hue[ii]) * R.brightness;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
