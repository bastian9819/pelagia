/**
 * "God mode": live sliders that write straight into the params uniform the
 * compute shaders read every tick, so changes take effect immediately. Only
 * runtime-safe parameters are exposed (ones that don't resize buffers or the
 * spatial grid). `idx` is the f32 index into the params buffer.
 */
export interface GodSpec {
  label: string;
  idx: number;
  min: number;
  max: number;
  step: number;
  value: number;
}

export interface GodPanel {
  panel: HTMLElement;
  toggle: HTMLButtonElement;
}

export function buildGodPanel(
  specs: GodSpec[],
  onChange: (idx: number, value: number) => void,
): GodPanel {
  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;right:12px;bottom:72px;width:250px;display:none;padding:12px 14px;' +
    'font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;' +
    'background:rgba(2,4,10,0.72);border:1px solid rgba(63,240,216,0.22);border-radius:10px;';

  const title = document.createElement('div');
  title.textContent = 'god mode';
  title.style.cssText = 'font-weight:600;letter-spacing:.1em;color:#3ff0d8;margin-bottom:10px;';
  panel.append(title);

  const inputs: { input: HTMLInputElement; valEl: HTMLElement; spec: GodSpec }[] = [];
  for (const spec of specs) {
    const row = document.createElement('div');
    row.style.cssText = 'margin-bottom:10px;';
    const head = document.createElement('div');
    head.style.cssText = 'display:flex;justify-content:space-between;opacity:.85;';
    const label = document.createElement('span');
    label.textContent = spec.label;
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
    panel.append(row);
    inputs.push({ input, valEl, spec });
  }

  const reset = document.createElement('button');
  reset.textContent = 'reset';
  reset.style.cssText =
    'margin-top:4px;padding:6px 12px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  reset.onclick = () => {
    for (const { input, valEl, spec } of inputs) {
      input.value = String(spec.value);
      valEl.textContent = fmt(spec.value);
      onChange(spec.idx, spec.value);
    }
  };
  panel.append(reset);

  const toggle = document.createElement('button');
  toggle.textContent = '⚙ god';
  toggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  toggle.onclick = () => {
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  };

  return { panel, toggle };
}

function fmt(v: number): string {
  return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
}
