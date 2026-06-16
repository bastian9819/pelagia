// Sense (nearest food via the grid) -> brain forward pass -> move.
// cellSize == perception radius, so a 3x3 cell block covers perception.
// Creatures only read the food grid and their own state, so the position
// update is safe in place (no creature-vs-creature reads in this spike).

const INPUT_SIZE: u32 = 8u;
const HIDDEN_SIZE: u32 = 10u;
const OUTPUT_SIZE: u32 = 2u;
const GENOME_SIZE: u32 = 112u;
const NONE: u32 = 0xffffffffu;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= GP.dims.w) {
    return;
  }
  let s = state[i];
  let W = GP.worldSize.x;
  let H = GP.worldSize.y;
  let cs = GP.worldSize.z;
  let cols = i32(GP.dims.x);
  let rows = i32(GP.dims.y);

  let bcx = i32(floor(s.x / cs));
  let bcy = i32(floor(s.y / cs));

  var bestD2 = 1.0e30;
  var bestIdx = NONE;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    var cy = (bcy + dy) % rows;
    if (cy < 0) { cy = cy + rows; }
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      var cx = (bcx + dx) % cols;
      if (cx < 0) { cx = cx + cols; }
      let cell = u32(cy) * GP.dims.x + u32(cx);
      let start = cellStart[cell];
      let end = cellStart[cell + 1u];
      for (var k = start; k < end; k = k + 1u) {
        let j = sortedIdx[k];
        let fp = foodPos[j];
        let ddx = wrapDelta(fp.x - s.x, W);
        let ddy = wrapDelta(fp.y - s.y, H);
        let d2 = ddx * ddx + ddy * ddy;
        if (d2 < bestD2) {
          bestD2 = d2;
          bestIdx = j;
        }
      }
    }
  }

  var inp: array<f32, INPUT_SIZE>;
  if (bestIdx != NONE && bestD2 <= cs * cs) {
    let fp = foodPos[bestIdx];
    let ddx = wrapDelta(fp.x - s.x, W);
    let ddy = wrapDelta(fp.y - s.y, H);
    let dist = sqrt(bestD2);
    let rel = atan2(ddy, ddx) - s.z;
    inp[0] = cos(rel);
    inp[1] = sin(rel);
    inp[2] = max(0.0, 1.0 - dist / cs);
    nearestOut[i] = i32(bestIdx);
  } else {
    inp[0] = 0.0;
    inp[1] = 0.0;
    inp[2] = 0.0;
    nearestOut[i] = -1;
  }
  inp[3] = 0.0;
  inp[4] = 0.0;
  inp[5] = 0.0;
  inp[6] = 0.5;
  inp[7] = s.w / GP.worldSize.w;

  var p = i * GENOME_SIZE;
  var hidden: array<f32, HIDDEN_SIZE>;
  for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
    var sum = 0.0;
    for (var k = 0u; k < INPUT_SIZE; k = k + 1u) {
      sum = sum + weights[p] * inp[k];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    hidden[h] = tanh(sum);
  }
  var out: array<f32, OUTPUT_SIZE>;
  for (var o = 0u; o < OUTPUT_SIZE; o = o + 1u) {
    var sum = 0.0;
    for (var h = 0u; h < HIDDEN_SIZE; h = h + 1u) {
      sum = sum + weights[p] * hidden[h];
      p = p + 1u;
    }
    sum = sum + weights[p];
    p = p + 1u;
    out[o] = tanh(sum);
  }

  let heading = s.z + out[0] * GP.motion.x;
  let speed = (out[1] + 1.0) * 0.5 * GP.worldSize.w;
  var x = s.x + cos(heading) * speed * GP.motion.y;
  var y = s.y + sin(heading) * speed * GP.motion.y;
  x = x - floor(x / W) * W;
  y = y - floor(y / H) * H;
  state[i] = vec4<f32>(x, y, heading, speed);
}
