/**
 * Phase 5: reproducible oceans via shareable URLs (wow #5).
 *
 * An ocean is fully determined by `(seed, n)` for its initial state and by
 * `(seed, god-mode params)` for its trajectory: the seed drives both the CPU-side
 * initialisation (positions, genomes, food) AND the per-tick GPU randomness
 * (mutation, food respawn), folded into the shader RNG via the params uniform.
 *
 * We serialize seed + N + god params into the URL hash so a link reproduces the
 * SAME initial ocean and the SAME rules. Honest caveat (R-002): on the same
 * device it's near-identical; across devices it evolves with the same *character*
 * but not bit-for-bit (GPU float drift + atomic ordering). We deliberately do NOT
 * fall back to the exact CPU oracle for playback — that would kill "thousands at
 * 60 fps", which is the whole point.
 *
 * The encode/decode functions are pure (no DOM) so they can be unit-tested in
 * Node; the thin wrappers below read `location` / write the clipboard.
 */

/** A fully-specified shareable ocean. */
export interface ShareState {
  /** u32 world seed (drives init + shader RNG). */
  seed: number;
  /** Requested creature slot count. */
  n: number;
  /** God-mode params as (params-uniform float index -> value) pairs. */
  params: ReadonlyArray<{ idx: number; value: number }>;
}

/** What we can recover from a hash; any field may be absent / malformed-away. */
export interface ParsedShare {
  seed?: number;
  n?: number;
  params?: { idx: number; value: number }[];
}

/** Bumped only if the hash grammar changes incompatibly. */
const HASH_VERSION = '1';

/** Short, lossless-enough number rendering (≤4 decimals, no trailing zeros). */
function fmtNum(v: number): string {
  return String(Math.round(v * 1e4) / 1e4);
}

/** Encode a share state to a `#...` URL fragment. Pure. */
export function encodeHash(state: ShareState): string {
  const parts = [`v=${HASH_VERSION}`, `s=${state.seed >>> 0}`, `n=${Math.floor(state.n)}`];
  if (state.params.length) {
    parts.push('g=' + state.params.map((p) => `${p.idx}:${fmtNum(p.value)}`).join(','));
  }
  return '#' + parts.join('&');
}

/**
 * Decode a `#...` URL fragment back into a (partial) share state. Pure and
 * defensive: unknown/garbage fields are dropped rather than thrown on. Returns
 * `null` if nothing usable was found.
 */
export function decodeHash(hash: string): ParsedShare | null {
  const raw = hash.replace(/^#/, '');
  if (!raw) return null;
  const q = new URLSearchParams(raw);
  const out: ParsedShare = {};

  const s = q.get('s');
  if (s !== null && /^\d+$/.test(s)) out.seed = Number(s) >>> 0;

  const n = q.get('n');
  if (n !== null && /^\d+$/.test(n)) {
    const nv = Math.floor(Number(n));
    if (nv >= 1) out.n = nv;
  }

  const g = q.get('g');
  if (g) {
    const params: { idx: number; value: number }[] = [];
    for (const pair of g.split(',')) {
      const [k, v] = pair.split(':');
      if (k === undefined || v === undefined) continue;
      const idx = Number(k);
      const value = Number(v);
      // Guard the index range (params uniform is 48 floats/u32) and finiteness;
      // the caller further restricts to known god-mode indices and clamps ranges.
      if (Number.isInteger(idx) && idx >= 0 && idx < 48 && Number.isFinite(value)) {
        params.push({ idx, value });
      }
    }
    if (params.length) out.params = params;
  }

  return Object.keys(out).length ? out : null;
}

/** Read the current ocean's share state from the address-bar hash. */
export function parseShareState(): ParsedShare | null {
  return decodeHash(location.hash);
}

/** Build a full shareable URL for the given ocean. */
export function buildShareUrl(state: ShareState): string {
  const url = new URL(location.href);
  url.hash = encodeHash(state);
  return url.toString();
}

/** Reflect the current ocean in the address bar (no navigation, no history spam). */
export function applyHash(state: ShareState): void {
  history.replaceState(null, '', buildShareUrl(state));
}

/**
 * Copy text to the clipboard. Uses the async Clipboard API when available
 * (secure contexts) and falls back to a hidden-textarea `execCommand` otherwise.
 * Returns whether the copy succeeded.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}
