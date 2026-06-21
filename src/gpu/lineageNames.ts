/**
 * Custom lineage names — a NAME layer on top of the functional lineage id.
 *
 * The lineage id (`bio.w`) is used by the simulation and UI for real work:
 * colour (hue from the id), slot reuse detection, speciation parent pointers and
 * the observatory watch-list. So it must NOT change. This module keeps a separate
 * `Map<id, name>` purely for display: the founder can christen a clade
 * ("Hunters", "Drifters") and see that name wherever the id used to show, with
 * `#id` kept as a dim secondary so the functional id is never lost.
 *
 * Persistence is per `${seed}` in localStorage, so names stick to the ocean they
 * describe (a shared seed restores the same lineages). No simulation state.
 */

const names = new Map<number, string>();
const listeners = new Set<() => void>();
let storageKey: string | null = null;

const MAX_NAME = 28; // keep labels short enough to sit beside the #id

function notify(): void {
  for (const fn of listeners) fn();
}

function persist(): void {
  if (!storageKey) return;
  try {
    const obj: Record<string, string> = {};
    for (const [id, name] of names) obj[id] = name;
    if (Object.keys(obj).length === 0) localStorage.removeItem(storageKey);
    else localStorage.setItem(storageKey, JSON.stringify(obj));
  } catch {
    // localStorage may be unavailable (private mode / quota) — names just won't persist.
  }
}

/**
 * Namespace persisted names by the world seed and load any saved ones. Called
 * once at sim init; a different seed (= different ocean) gets its own name set.
 */
export function setLineageSeed(seed: number): void {
  storageKey = `pelagia:lineageNames:${seed >>> 0}`;
  names.clear();
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) {
      const obj = JSON.parse(raw) as Record<string, string>;
      for (const k of Object.keys(obj)) {
        const id = Number(k);
        const v = obj[k];
        if (Number.isFinite(id) && typeof v === 'string' && v.length > 0) names.set(id, v);
      }
    }
  } catch {
    // ignore corrupt/blocked storage
  }
  notify();
}

/** The custom name for a lineage, or undefined if it has none. */
export function getLineageName(id: number): string | undefined {
  return names.get(id);
}

/** Set (or clear, with an empty string) a lineage's custom name. */
export function setLineageName(id: number, name: string): void {
  const trimmed = name.trim().slice(0, MAX_NAME);
  if (trimmed.length === 0) {
    if (!names.delete(id)) return;
  } else {
    if (names.get(id) === trimmed) return;
    names.set(id, trimmed);
  }
  persist();
  notify();
}

/** Display string for a lineage: its custom name if any, else `#id`. */
export function displayLineage(id: number): string {
  return names.get(id) ?? `#${id}`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Rich label HTML: a named lineage shows its name in bold with a dim `#id`
 * secondary; an unnamed one shows `<b>#id</b>` exactly as before. Safe to drop
 * into innerHTML (the user-entered name is escaped).
 */
export function lineageLabelHtml(id: number): string {
  const name = names.get(id);
  if (name === undefined) return `<b>#${id}</b>`;
  return (
    `<b>${escapeHtml(name)}</b>` +
    `<span style="opacity:.5;font-weight:400;font-size:.85em;margin-left:5px">#${id}</span>`
  );
}

/** Re-render hook: fires whenever any name (or the seed) changes. */
export function onLineageNamesChange(fn: () => void): void {
  listeners.add(fn);
}
