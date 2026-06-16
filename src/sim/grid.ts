import { wrap, toroidalDistSq } from '../core/space.js';

/**
 * Uniform spatial grid for O(N) neighbour queries on a toroidal world.
 *
 * The build uses a counting sort — count items per cell, prefix-sum into cell
 * start offsets, then scatter item indices into a sorted array — which is the
 * exact technique the GPU port will use (atomic counters + prefix scan in a
 * compute shader). Keeping the CPU reference structurally identical to the
 * planned shader means the Phase 1 port is a near-mechanical translation, and
 * this implementation stays a faithful oracle for it.
 *
 * Cell size should be roughly the perception radius so a query only needs to
 * scan a small block of neighbouring cells.
 */
export class SpatialGrid {
  readonly width: number;
  readonly height: number;
  readonly cellSize: number;
  readonly cols: number;
  readonly rows: number;
  readonly numCells: number;

  private readonly counts: Int32Array;
  private readonly cellStart: Int32Array; // length numCells + 1
  private sorted: Int32Array;
  private capacity = 0;

  // References to the arrays indexed by the last build().
  private itemX: Float32Array = new Float32Array(0);
  private itemY: Float32Array = new Float32Array(0);
  private itemCount = 0;

  constructor(width: number, height: number, cellSize: number) {
    this.width = width;
    this.height = height;
    this.cellSize = cellSize;
    this.cols = Math.max(1, Math.ceil(width / cellSize));
    this.rows = Math.max(1, Math.ceil(height / cellSize));
    this.numCells = this.cols * this.rows;
    this.counts = new Int32Array(this.numCells);
    this.cellStart = new Int32Array(this.numCells + 1);
    this.sorted = new Int32Array(0);
  }

  private cellOf(x: number, y: number): number {
    const cx = Math.min(this.cols - 1, Math.floor(wrap(x, this.width) / this.cellSize));
    const cy = Math.min(this.rows - 1, Math.floor(wrap(y, this.height) / this.cellSize));
    return cy * this.cols + cx;
  }

  /** Rebuild the grid over the first `count` items of the parallel x/y arrays. */
  build(x: Float32Array, y: Float32Array, count: number): void {
    this.itemX = x;
    this.itemY = y;
    this.itemCount = count;

    if (count > this.capacity) {
      this.sorted = new Int32Array(count);
      this.capacity = count;
    }

    this.counts.fill(0);
    for (let i = 0; i < count; i++) this.counts[this.cellOf(x[i]!, y[i]!)]!++;

    // Prefix sum into cell start offsets.
    let running = 0;
    for (let c = 0; c < this.numCells; c++) {
      this.cellStart[c] = running;
      running += this.counts[c]!;
    }
    this.cellStart[this.numCells] = running;

    // Scatter item indices into their cell buckets (counts reused as cursors).
    // Reset counts to the start offsets, then advance as we place each item.
    for (let c = 0; c < this.numCells; c++) this.counts[c] = this.cellStart[c]!;
    for (let i = 0; i < count; i++) {
      const c = this.cellOf(x[i]!, y[i]!);
      this.sorted[this.counts[c]!] = i;
      this.counts[c]!++;
    }
  }

  /**
   * Invoke `cb(itemIndex, distSq)` for every indexed item within `radius` of
   * (qx, qy), honouring toroidal wrap. Allocation-free.
   */
  forEachNeighbor(
    qx: number,
    qy: number,
    radius: number,
    cb: (index: number, distSq: number) => void,
  ): void {
    if (this.itemCount === 0) return;
    const r2 = radius * radius;
    const cr = Math.max(0, Math.ceil(radius / this.cellSize));
    const baseCx = Math.min(this.cols - 1, Math.floor(wrap(qx, this.width) / this.cellSize));
    const baseCy = Math.min(this.rows - 1, Math.floor(wrap(qy, this.height) / this.cellSize));

    // If the query block would cover every column/row, scan each once to avoid
    // visiting the same wrapped cell twice (small worlds).
    const spanX = Math.min(2 * cr + 1, this.cols);
    const spanY = Math.min(2 * cr + 1, this.rows);

    for (let oy = 0; oy < spanY; oy++) {
      const cy = spanY === this.rows ? oy : wrapIndex(baseCy + oy - cr, this.rows);
      for (let ox = 0; ox < spanX; ox++) {
        const cx = spanX === this.cols ? ox : wrapIndex(baseCx + ox - cr, this.cols);
        const cell = cy * this.cols + cx;
        const end = this.cellStart[cell + 1]!;
        for (let s = this.cellStart[cell]!; s < end; s++) {
          const idx = this.sorted[s]!;
          const d2 = toroidalDistSq(
            qx,
            qy,
            this.itemX[idx]!,
            this.itemY[idx]!,
            this.width,
            this.height,
          );
          if (d2 <= r2) cb(idx, d2);
        }
      }
    }
  }

  /**
   * Index of the nearest indexed item within `radius` of (qx, qy), or -1 if
   * none. Ties break toward the lower item index, so the result is independent
   * of cell iteration order (and therefore of swap-remove reordering).
   */
  findNearest(qx: number, qy: number, radius: number): number {
    let bestIdx = -1;
    let bestD2 = Infinity;
    this.forEachNeighbor(qx, qy, radius, (idx, d2) => {
      if (d2 < bestD2 || (d2 === bestD2 && idx < bestIdx)) {
        bestD2 = d2;
        bestIdx = idx;
      }
    });
    return bestIdx;
  }
}

/** Wrap an integer cell index into [0, n). */
function wrapIndex(i: number, n: number): number {
  const m = i % n;
  return m < 0 ? m + n : m;
}
