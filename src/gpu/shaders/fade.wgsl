// Fullscreen fade: alpha-blends a dark wash over the accumulation texture each
// frame so creature glows leave fading motion trails (the bioluminescent look).

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  // RGB ~ deep ocean; alpha = per-frame fade rate (higher = shorter trails).
  // Shorter trails so the creature bodies read clearly (not just streaks).
  return vec4<f32>(0.006, 0.013, 0.03, 0.18);
}
