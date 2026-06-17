/**
 * Lineage explorer: ranks dominant lineages and characterises each one's neural
 * policy by probing a representative brain with canonical stimuli (food ahead /
 * to the side / absent) using the CPU forward pass. Answers "what neural
 * decisions made this lineage dominate?". Foraging only — aggression/predation
 * arrives in Phase 6 (creatures can't eat each other yet).
 */
import { forward, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE } from '../sim/brain.js';
import { t, onLang } from './i18n.js';

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
  inp[6] = 0.5;
  inp[7] = 0.4;
  forward(genome, 0, inp, hid, out);
  return { turn: out[0]!, thrust: (out[1]! + 1) / 2 };
}

/** Behavioural profile of a genome as a translation key + a fast/slow flag. */
export function characterizeGenome(genome: Float32Array): { descKey: string; fast: boolean } {
  const ahead = probe(genome, 1, 0, 0.8);
  const left = probe(genome, 0, 1, 0.8);
  const right = probe(genome, 0, -1, 0.8);
  const none = probe(genome, 0, 0, 0);
  const forage = ahead.thrust;
  const steer = (left.turn - right.turn) / 2;
  const cruise = none.thrust;

  let descKey: string;
  if (steer > 0.25 && forage > 0.45) descKey = 'desc_chase';
  else if (steer > 0.25) descKey = 'desc_steer';
  else if (forage > 0.55 && cruise > 0.5) descKey = 'desc_straight';
  else if (steer < -0.2) descKey = 'desc_away';
  else descKey = 'desc_erratic';
  return { descKey, fast: (forage + cruise) / 2 > 0.55 };
}

export interface LineageRow {
  lineage: number;
  hue: number;
  count: number;
  trend: number;
  descKey: string;
  fast: boolean;
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
  title.style.cssText = 'font-weight:600;letter-spacing:.1em;color:#3ff0d8;margin-bottom:8px;';
  const list = document.createElement('div');
  panel.append(title, list);

  const toggle = document.createElement('button');
  toggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  toggle.onclick = () => {
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  };

  let lastRows: LineageRow[] = [];
  function render(): void {
    title.textContent = t('dominantLineages');
    toggle.textContent = '🧬 ' + t('lineages');
    if (lastRows.length === 0) {
      list.innerHTML = `<div style="opacity:.5">${t('sampling')}</div>`;
      return;
    }
    list.innerHTML = lastRows
      .map((r) => {
        const c = `hsl(${Math.round(r.hue * 360)}, 90%, 62%)`;
        const arrow = r.trend > 1 ? '▲' : r.trend < -1 ? '▼' : '—';
        const ac = r.trend > 1 ? '#3ff0d8' : r.trend < -1 ? '#ff5aa6' : 'rgba(207,232,255,.5)';
        const desc = `${t(r.descKey)} · ${t(r.fast ? 'fast' : 'slow')}`;
        return (
          `<div style="margin-bottom:9px;line-height:1.45">` +
          `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>` +
          `<b>#${r.lineage}</b> · ${r.count} <span style="color:${ac}">${arrow}</span>` +
          `<div style="opacity:.7;margin-left:16px">${desc}</div></div>`
        );
      })
      .join('');
  }
  render();
  onLang(render);

  return {
    panel,
    toggle,
    update(rows) {
      lastRows = rows;
      render();
    },
  };
}
