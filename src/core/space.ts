/**
 * Toroidal (wrap-around) 2D space helpers.
 *
 * The ocean wraps at the edges so there are no walls to pile up against — a
 * creature leaving the right edge reappears on the left. Distances and
 * directions must therefore take the shortest path across the seam.
 *
 * These are allocation-free scalar helpers; hot loops (sensors, the spatial
 * grid) call them directly rather than passing vectors around.
 */

/** Wrap a coordinate into [0, size). */
export function wrap(value: number, size: number): number {
  const m = value % size;
  return m < 0 ? m + size : m;
}

/**
 * Shortest signed delta along one axis on a torus of the given size.
 * Result is in (-size/2, size/2].
 */
export function wrapDelta(delta: number, size: number): number {
  const half = size * 0.5;
  if (delta > half) return delta - size;
  if (delta < -half) return delta + size;
  return delta;
}

/** Squared toroidal distance between (ax, ay) and (bx, by). */
export function toroidalDistSq(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  width: number,
  height: number,
): number {
  const dx = wrapDelta(bx - ax, width);
  const dy = wrapDelta(by - ay, height);
  return dx * dx + dy * dy;
}
