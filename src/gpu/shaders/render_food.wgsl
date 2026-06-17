// Food as glowing motes (additive). Dead food (x < 0) is culled. Two types by
// index: indices [0, bigCount) are rare "big food" (gold, large); the rest are
// abundant plankton (pale teal, small).

struct RParams {
  view: vec4<f32>,
  sizeWorld: f32,
  brightness: f32,
  _p0: f32,      // colorMode (used by the creature shader; ignored here)
  bigCount: f32, // number of big-food slots — indices [0, bigCount)
};

@group(0) @binding(0) var<uniform> R: RParams;
@group(0) @binding(1) var<storage, read> foodPos: array<vec2<f32>>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) big: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );
  var out: VSOut;
  let p = foodPos[ii];
  if (p.x < 0.0) {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.big = 0.0;
    return out;
  }
  let big = select(0.0, 1.0, f32(ii) < R.bigCount);
  let q = quad[vi];
  let world = p + q * (R.sizeWorld * select(0.45, 1.4, big > 0.5));
  out.pos = vec4<f32>(world.x * R.view.x + R.view.z, world.y * R.view.y + R.view.w, 0.0, 1.0);
  out.uv = q;
  out.big = big;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  let glow = smoothstep(1.0, 0.0, d);
  // Pale teal plankton vs bright gold big-food.
  let plankton = vec3<f32>(0.25, 0.7, 0.6) * 0.5;
  let bigCol = vec3<f32>(1.0, 0.78, 0.28) * 1.4;
  let col = select(plankton, bigCol, in.big > 0.5);
  return vec4<f32>(col * glow * glow, 1.0);
}
