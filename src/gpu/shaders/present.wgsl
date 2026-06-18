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
  c: vec4<f32>,    // frame, currentStrength, currentViz (0 = off), _
  e: vec4<f32>,    // pheroBase, pheroViz (0 = off), _, _
};
@group(0) @binding(2) var<uniform> R: RParams;
@group(0) @binding(3) var<storage, read> gridDataRO: array<u32>; // pheromone field
const PHERO_RES_P: u32 = 128u;

// Ocean current at a world position (mirror of life_common's currentAt) so the
// flow can be drawn as animated streaks.
fn currentAt(x: f32, y: f32, frame: f32, world: f32) -> vec2<f32> {
  let TAU = 6.2831853;
  let u = (x / world) * TAU;
  let w = (y / world) * TAU;
  let t = frame * 0.0008;
  let vx = cos(u + t) * cos(w) + 0.5 * cos(2.0 * u - t) * cos(2.0 * w);
  let vy = -sin(u + t) * sin(w) - 0.5 * sin(2.0 * u - t) * sin(2.0 * w);
  return vec2<f32>(vx, vy);
}

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
  // Current reveal: animated streaks aligned with the flow field — moving bands
  // perpendicular to the local flow direction, brighter where the current is fast.
  if (R.c.z > 0.0 && R.c.y > 0.0) {
    let clipx = in.uv.x * 2.0 - 1.0;
    let clipy = 1.0 - in.uv.y * 2.0;
    let wx = (clipx - R.view.z) / R.view.x;
    let wy = (clipy - R.view.w) / R.view.y;
    let flow = currentAt(wx, wy, R.c.x, R.b.w);
    let mag = length(flow);
    let dir = flow / max(mag, 0.001);
    let along = (wx * dir.x + wy * dir.y) * 0.06 - R.c.x * 0.05;
    let streak = smoothstep(0.55, 1.0, sin(along * 6.2831853));
    col = col + vec3<f32>(0.16, 0.55, 0.75) * streak * mag * R.c.y * 0.16;
  }
  // Pheromone reveal: the trails creatures lay glow as faint green paths.
  if (R.e.y > 0.0) {
    let clipx = in.uv.x * 2.0 - 1.0;
    let clipy = 1.0 - in.uv.y * 2.0;
    let wx = (clipx - R.view.z) / R.view.x;
    let wy = (clipy - R.view.w) / R.view.y;
    let world = R.b.w;
    let fx = u32(clamp(wx / world, 0.0, 0.99999) * f32(PHERO_RES_P));
    let fy = u32(clamp(wy / world, 0.0, 0.99999) * f32(PHERO_RES_P));
    let lvl = f32(gridDataRO[u32(R.e.x) + fy * PHERO_RES_P + fx]);
    let p = clamp(lvl / 60000.0, 0.0, 1.0);
    col = col + vec3<f32>(0.10, 0.45, 0.20) * p;
  }
  return vec4<f32>(col, 1.0);
}
