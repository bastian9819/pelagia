/**
 * Shared rename popover for lineages — a single floating input reused from every
 * panel (lineages list, observatory rows/watch-list, history legend, cladogram
 * nodes). Centralising it here means one rename UX everywhere and avoids managing
 * an inline input inside lists that re-render every frame. Saving flows through
 * setLineageName, whose onLineageNamesChange hook re-labels all panels live.
 */
import { getLineageName, setLineageName } from './lineageNames.js';
import { icon } from './icons.js';
import { t } from './i18n.js';

let pop: HTMLDivElement | null = null;
let onOutside: ((e: MouseEvent) => void) | null = null;

export function closeLineageRename(): void {
  if (onOutside) {
    document.removeEventListener('mousedown', onOutside, true);
    onOutside = null;
  }
  pop?.remove();
  pop = null;
}

/**
 * Open the rename popover for a lineage near (clientX, clientY). Enter or the
 * check saves; Escape, the ✕ or an outside click cancels. Keystrokes are kept
 * off the global shortcuts.
 */
export function openLineageRename(id: number, clientX: number, clientY: number): void {
  closeLineageRename();
  const el = document.createElement('div');
  el.className = 'pg-panel';
  el.style.cssText =
    'position:fixed;z-index:2000;padding:11px 12px;display:flex;flex-direction:column;gap:9px;width:236px;';

  const head = document.createElement('div');
  head.className = 'pg-eyebrow';
  head.style.cssText = 'display:flex;align-items:center;gap:6px;';
  head.innerHTML = `${t('nameLineage')} <span style="opacity:.6">#${id}</span>`;

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;align-items:center;gap:6px;';
  const input = document.createElement('input');
  input.type = 'text';
  input.value = getLineageName(id) ?? '';
  input.placeholder = `#${id}`;
  input.maxLength = 28;
  input.setAttribute('aria-label', t('nameLineage'));
  input.style.cssText =
    'flex:1;min-width:0;background:var(--surface-2);border:1px solid var(--border-2);' +
    'border-radius:6px;color:var(--ink);font:12.5px var(--font-ui);padding:5px 8px;outline:none;';
  const save = document.createElement('button');
  save.className = 'pg-btn pg-iconbtn';
  save.style.cssText = 'width:28px;height:28px;flex:none;';
  save.innerHTML = icon('check', 15);
  save.setAttribute('aria-label', t('save'));
  row.append(input, save);

  const hint = document.createElement('div');
  hint.style.cssText = 'font-size:10.5px;color:var(--ink-faint);';
  hint.textContent = t('renameHint');

  el.append(head, row, hint);
  document.body.append(el);
  pop = el; // track it so commit/cancel/outside-click can remove it

  // Position near the click, clamped to the viewport.
  const W = 236;
  const H = el.offsetHeight || 96;
  const x = Math.max(8, Math.min(clientX, window.innerWidth - W - 8));
  const y = Math.max(8, Math.min(clientY + 10, window.innerHeight - H - 8));
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;

  const commit = (): void => {
    setLineageName(id, input.value);
    closeLineageRename();
  };
  input.onkeydown = (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') commit();
    else if (e.key === 'Escape') closeLineageRename();
  };
  save.onclick = commit;

  onOutside = (e: MouseEvent): void => {
    if (!el.contains(e.target as Node)) closeLineageRename();
  };
  // capture so it runs before other handlers; defer to skip the opening click
  setTimeout(() => {
    if (onOutside) document.addEventListener('mousedown', onOutside, true);
  }, 0);

  input.focus();
  input.select();
}
