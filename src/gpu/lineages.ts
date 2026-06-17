/**
 * Lineage explorer: ranks the dominant lineages and characterises each one's
 * neural policy in plain language by probing a representative brain with
 * canonical stimuli (food ahead / to the side / absent) using the same CPU
 * forward pass as the reference simulation. Answers "what neural decisions made
 * this lineage dominate?".
 *
 * Aggression / predation is intentionally out of scope here — creatures can't
 * eat each other yet (that arrives in Phase 6); this characterises foraging.
 */
import { forward, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE } from '../sim/brain.js';

const inp = new Float32Array(INPUT_SIZE);
const hid = new Float32Array(HIDDEN_SIZE);
const out = new Float32Array(OUTPUT_SIZE);

function probe(
  genome: Float32Array,
  foodCos: number,
  foodSin: number,
  foodProx: number,
): { turn: number; thrust: number } {
  inp.fill(0);
  inp[0] = foodCos;
  inp[1] = foodSin;
  inp[2] = foodProx;
  inp[6] = 0.5; // mid energy
  inp[7] = 0.4; // mid speed
  forward(genome, 0, inp, hid, out);
  return { turn: out[0]!, thrust: (out[1]! + 1) / 2 };
}

/** One-line behavioural profile of a genome, from canonical stimulus responses. */
export function characterizeGenome(genome: Float32Array): string {
  const ahead = probe(genome, 1, 0, 0.8);
  const left = probe(genome, 0, 1, 0.8);
  const right = probe(genome, 0, -1, 0.8);
  const none = probe(genome, 0, 0, 0);

  const forage = ahead.thrust; // thrust when food is dead ahead
  const steer = (left.turn - right.turn) / 2; // > 0 = turns toward food
  const cruise = none.thrust; // movement with no food in sight

  let desc: string;
  if (steer > 0.25 && forage > 0.45) desc = 'chases food head-on';
  else if (steer > 0.25) desc = 'steers toward food (cautious)';
  else if (forage > 0.55 && cruise > 0.5) desc = 'fast straight-swimmer';
  else if (steer < -0.2) desc = 'turns away from food';
  else desc = 'erratic / undirected';
  return `${desc} · ${(forage + cruise) / 2 > 0.55 ? 'fast' : 'slow'}`;
}

export interface LineageRow {
  lineage: number;
  hue: number;
  count: number;
  trend: number; // signed change since last sample
  desc: string;
}

export interface LineagePanel {
  panel: HTMLElement;
  toggle: HTMLButtonElement;
  update(rows: LineageRow[]): void;
}

export function buildLineagePanel(): LineagePanel {
  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;left:12px;top:232px;width:248px;max-height:50vh;overflow:auto;display:none;' +
    'padding:12px 14px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;' +
    'background:rgba(2,4,10,0.66);border:1px solid rgba(63,240,216,0.18);border-radius:10px;';
  const title = document.createElement('div');
  title.textContent = 'dominant lineages';
  title.style.cssText = 'font-weight:600;letter-spacing:.1em;color:#3ff0d8;margin-bottom:8px;';
  const list = document.createElement('div');
  panel.append(title, list);

  const toggle = document.createElement('button');
  toggle.textContent = '🧬 lineages';
  toggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  toggle.onclick = () => {
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  };

  function update(rows: LineageRow[]): void {
    list.innerHTML = rows
      .map((r) => {
        const c = `hsl(${Math.round(r.hue * 360)}, 90%, 62%)`;
        const arrow = r.trend > 1 ? '▲' : r.trend < -1 ? '▼' : '—';
        const arrowColor =
          r.trend > 1 ? '#3ff0d8' : r.trend < -1 ? '#ff5aa6' : 'rgba(207,232,255,.5)';
        return (
          `<div style="margin-bottom:9px;line-height:1.45">` +
          `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>` +
          `<b>#${r.lineage}</b> · ${r.count} <span style="color:${arrowColor}">${arrow}</span>` +
          `<div style="opacity:.7;margin-left:16px">${r.desc}</div>` +
          `</div>`
        );
      })
      .join('');
    if (rows.length === 0) list.innerHTML = '<div style="opacity:.5">sampling…</div>';
  }

  return { panel, toggle, update };
}
