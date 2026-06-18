/**
 * "God mode": live sliders that write straight into the params uniform the
 * compute shaders read every tick, so changes apply immediately. Only
 * runtime-safe parameters are exposed (no buffer/grid resize). `idx` is the f32
 * index into the params buffer; `labelKey`/`group` are i18n keys. Sliders are
 * grouped into collapsible categories so the (now large) panel stays navigable.
 */
import { t, onLang } from './i18n.js';

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
  panel.style.cssText =
    'position:fixed;right:12px;bottom:72px;width:258px;max-height:80vh;overflow:auto;display:none;' +
    'padding:12px 14px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;' +
    'background:rgba(2,4,10,0.72);border:1px solid rgba(63,240,216,0.22);border-radius:10px;';

  const title = document.createElement('div');
  title.style.cssText = 'font-weight:600;letter-spacing:.1em;color:#3ff0d8;margin-bottom:10px;';
  panel.append(title);

  // Slot for callers to inject extra controls (toggles, presets, dice) above the
  // sliders. Populated by gpuSim after construction.
  const extras = document.createElement('div');
  panel.append(extras);

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
      'display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;margin:6px 0 6px;' +
      'color:#7fe9d8;letter-spacing:.06em;font-size:11px;';
    const caret = document.createElement('span');
    caret.textContent = '▾';
    const headLabel = document.createElement('span');
    header.append(caret, headLabel);

    const body = document.createElement('div');
    header.onclick = () => {
      const open = body.style.display !== 'none';
      body.style.display = open ? 'none' : 'block';
      caret.textContent = open ? '▸' : '▾';
    };
    sections.push({ headEl: headLabel, key: group, caret });

    for (const spec of byGroup.get(group)!) {
      const row = document.createElement('div');
      row.style.cssText = 'margin-bottom:10px;';
      const head = document.createElement('div');
      head.style.cssText = 'display:flex;justify-content:space-between;opacity:.85;';
      const label = document.createElement('span');
      const valEl = document.createElement('span');
      valEl.style.color = '#3ff0d8';
      valEl.textContent = fmt(spec.value);
      head.append(label, valEl);
      const input = document.createElement('input');
      input.type = 'range';
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
      input.value = String(spec.value);
      input.style.cssText = 'width:100%;accent-color:#3ff0d8;cursor:pointer;';
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

    panel.append(header, body);
  }

  const reset = document.createElement('button');
  reset.style.cssText =
    'margin-top:4px;padding:6px 12px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  reset.onclick = () => {
    for (const { input, valEl, spec } of inputs.values()) {
      input.value = String(spec.value);
      valEl.textContent = fmt(spec.value);
      onChange(spec.idx, spec.value);
    }
  };
  panel.append(reset);

  const toggle = document.createElement('button');
  toggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  toggle.onclick = () => {
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  };

  function relabel(): void {
    title.textContent = t('godMode');
    reset.textContent = t('reset');
    toggle.textContent = '⚙ ' + t('god');
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
