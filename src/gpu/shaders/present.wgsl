// Fullscreen present: sample the HDR accumulation texture, tonemap, output to
// the swapchain. Reinhard + mild gamma keeps the additive glow from clipping.
// Optionally paints a faint thermal-biome tint behind the ocean (field reveal).

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
// Shared render UBO; we only read the view transform + the field-tint fields.
struct RParams {
  view: vec4<f32>, // sx, sy, ox, oy (world -> clip)
  a: vec4<f32>,    // sizeWorld, brightness, colorMode, bigCount
  b: vec4<f32>,    // hl.x, hl.y, fieldTint (0 = off), world
  c: vec4<f32>,    // frame, _, _, _
};
@group(0) @binding(2) var<uniform> R: RParams;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Water temperature at a world position (mirror of life_common's tempAt) so the
// background can reveal the thermal biomes the creatures live in.
fn tempAt(x: f32, y: f32, frame: f32, world: f32) -> f32 {
  let TAU = 6.2831853;
  let drift = frame * 0.0004;
  return clamp(0.6 * cos((y / world) * TAU + drift) + 0.4 * sin((x / world) * TAU), -1.0, 1.0);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var out: VSOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  var uv = p[vi] * 0.5 + 0.5;
  uv.y = 1.0 - uv.y;
  out.uv = uv;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let hdr = textureSample(tex, samp, in.uv).rgb;
  let mapped = hdr / (hdr + vec3<f32>(1.0)); // Reinhard
  var col = pow(mapped, vec3<f32>(0.85));
  // Field reveal: tint the background by the temperature field (cold=blue,
  // warm=red), so the otherwise-invisible thermal biomes become visible.
  let tint = R.b.z;
  if (tint > 0.0) {
    let clipx = in.uv.x * 2.0 - 1.0;
    let clipy = 1.0 - in.uv.y * 2.0;
    let wx = (clipx - R.view.z) / R.view.x;
    let wy = (clipy - R.view.w) / R.view.y;
    let temp = tempAt(wx, wy, R.c.x, R.b.w);
    let cold = vec3<f32>(0.04, 0.10, 0.40);
    let warm = vec3<f32>(0.50, 0.13, 0.05);
    col = col + mix(cold, warm, (temp + 1.0) * 0.5) * tint;
  }
  return vec4<f32>(col, 1.0);
}
