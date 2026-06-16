// Fullscreen present: sample the HDR accumulation texture, tonemap, output to
// the swapchain. Reinhard + mild gamma keeps the additive glow from clipping.

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

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
  return vec4<f32>(pow(mapped, vec3<f32>(0.85)), 1.0);
}
