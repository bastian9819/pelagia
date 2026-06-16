/**
 * Food pellets, stored as a structure-of-arrays for a clean GPU port later.
 *
 * Active pellets are packed contiguously in [0, count). Removal is an O(1)
 * swap-remove (move the last pellet into the freed slot). This reorders the
 * array, so any "nearest food" query that could tie must break ties
 * deterministically by something stable (see the sensor code), not by array
 * position — otherwise reproducibility would depend on eat order.
 */
export class FoodField {
  readonly capacity: number;
  readonly x: Float32Array;
  readonly y: Float32Array;
  count = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.x = new Float32Array(capacity);
    this.y = new Float32Array(capacity);
  }

  /** Add a pellet. Returns false (no-op) if the field is at capacity. */
  add(x: number, y: number): boolean {
    if (this.count >= this.capacity) return false;
    this.x[this.count] = x;
    this.y[this.count] = y;
    this.count++;
    return true;
  }

  /** Swap-remove the pellet at `index`. */
  removeSwap(index: number): void {
    const last = this.count - 1;
    if (index < 0 || index > last) return;
    this.x[index] = this.x[last]!;
    this.y[index] = this.y[last]!;
    this.count--;
  }

  clear(): void {
    this.count = 0;
  }
}
