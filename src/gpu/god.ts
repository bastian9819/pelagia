/**
 * "God mode": live sliders that write straight into the params uniform the
 * compute shaders read every tick, so changes apply immediately. Only
 * runtime-safe parameters are exposed (no buffer/grid resize). `idx` is the f32
 * index into the params buffer; `labelKey`/`group` are i18n keys. Sliders are
 * grouped into collapsible categories so the (now large) panel stays navigable.
 *
 * The panel is anchored to the LEFT rail (below the stats HUD, above the brush
 * dock) so it never overlaps the brain inspector, which lives on the right.
 */
import { t, onLang } from './i18n.js';
import { icon } from './icons.js';
import { makeDraggable, mkPanelHeader } from './ui.js';

export interface GodSpec {
  labelKey: string;
  /** i18n key of the collapsible category this slider lives under. */
  group: string;
  idx: number;
  min: number;
  max: number;
  step: number;
  value: number;
}

export interface GodPanel {
  panel: HTMLElement;
  toggle: HTMLButtonElement;
  /** Container above the sliders for extra controls (toggles, presets, dice). */
  extras: HTMLElement;
  /** Current live slider values as (params index -> value) pairs (for sharing). */
  getValues(): { idx: number; value: number }[];
  /** Push a value into a slider (updates the thumb + label) without firing onChange. */
  setValue(idx: number, value: number): void;
}

export function buildGodPanel(
  specs: GodSpec[],
  onChange: (idx: number, value: number) => void,
): GodPanel {
  const panel = document.createElement('div');
  panel.className = 'pg-panel';
  panel.style.cssText =
    'position:fixed;left:14px;top:236px;width:266px;max-height:calc(100vh - 60px);' +
    'display:none;flex-direction:column;overflow:hidden;padding:14px 15px;z-index:10;';

  // Draggable header with title + close (×); the body below scrolls independently.
  const { header: phead, title } = mkPanelHeader(() => (panel.style.display = 'none'));
  panel.append(phead);
  makeDraggable(panel, phead);

  const scroll = document.createElement('div');
  scroll.style.cssText = 'overflow:auto;flex:1;';
  panel.append(scroll);

  // Slot for callers to inject extra controls (toggles, presets, dice) above the
  // sliders. Populated by gpuSim after construction.
  const extras = document.createElement('div');
  scroll.append(extras);

  const labels: { el: HTMLElement; key: string }[] = [];
  const inputs = new Map<number, { input: HTMLInputElement; valEl: HTMLElement; spec: GodSpec }>();
  const sections: { headEl: HTMLElement; key: string; caret: HTMLElement }[] = [];

  // Group specs by category, preserving first-seen order.
  const order: string[] = [];
  const byGroup = new Map<string, GodSpec[]>();
  for (const spec of specs) {
    if (!byGroup.has(spec.group)) {
      byGroup.set(spec.group, []);
      order.push(spec.group);
    }
    byGroup.get(spec.group)!.push(spec);
  }

  for (const group of order) {
    const header = document.createElement('div');
    header.style.cssText =
      'display:flex;align-items:center;gap:7px;cursor:pointer;user-select:none;margin:14px 0 9px;';
    const caret = document.createElement('span');
    caret.style.cssText = 'color:var(--ink-faint);transition:transform .12s ease;display:flex;';
    caret.innerHTML = icon('step', 12); // chevrons, rotated 90° to point down when open
    caret.style.transform = 'rotate(90deg)';
    const headLabel = document.createElement('span');
    headLabel.className = 'pg-eyebrow';
    header.append(caret, headLabel);

    const body = document.createElement('div');
    header.onclick = () => {
      const open = body.style.display !== 'none';
      body.style.display = open ? 'none' : 'block';
      caret.style.transform = open ? 'rotate(0deg)' : 'rotate(90deg)';
    };
    sections.push({ headEl: headLabel, key: group, caret });

    for (const spec of byGroup.get(group)!) {
      const row = document.createElement('div');
      row.style.cssText = 'margin-bottom:13px;';
      const head = document.createElement('div');
      head.style.cssText = 'display:flex;justify-content:space-between;align-items:baseline;';
      const label = document.createElement('span');
      label.style.cssText = 'font-size:12px;color:var(--ink-dim);';
      const valEl = document.createElement('span');
      valEl.style.cssText = 'color:var(--glow-cyan);font:600 12px var(--font-mono);';
      valEl.textContent = fmt(spec.value);
      head.append(label, valEl);
      const input = document.createElement('input');
      input.type = 'range';
      input.className = 'pg-range';
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
      input.value = String(spec.value);
      input.style.marginTop = '7px';
      input.addEventListener('input', () => {
        const v = Number(input.value);
        valEl.textContent = fmt(v);
        onChange(spec.idx, v);
      });
      row.append(head, input);
      body.append(row);
      labels.push({ el: label, key: spec.labelKey });
      inputs.set(spec.idx, { input, valEl, spec });
    }

    scroll.append(header, body);
  }

  const reset = document.createElement('button');
  reset.className = 'pg-btn pg-row';
  reset.style.marginTop = '14px';
  reset.onclick = () => {
    for (const { input, valEl, spec } of inputs.values()) {
      input.value = String(spec.value);
      valEl.textContent = fmt(spec.value);
      onChange(spec.idx, spec.value);
    }
  };
  scroll.append(reset);

  const toggle = document.createElement('button');
  toggle.className = 'pg-btn';

  toggle.onclick = () => {
    panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
  };

  function relabel(): void {
    title.textContent = t('godMode');
    reset.innerHTML = icon('restart', 16) + `<span>${t('reset')}</span>`;
    toggle.innerHTML = icon('sliders', 16) + `<span>${t('god')}</span>`;
    for (const { el, key } of labels) el.textContent = t(key);
    for (const { headEl, key } of sections) headEl.textContent = t(key);
  }
  relabel();
  onLang(relabel);

  return {
    panel,
    toggle,
    extras,
    getValues: () =>
      [...inputs.values()].map(({ input, spec }) => ({
        idx: spec.idx,
        value: Number(input.value),
      })),
    setValue: (idx, value) => {
      const e = inputs.get(idx);
      if (!e) return;
      e.input.value = String(value);
      e.valEl.textContent = fmt(value);
    },
  };
}

function fmt(v: number): string {
  return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
}
