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

/** Continuous behavioural traits of a genome + a headline label. */
export interface LineageTraits {
  descKey: string;
  fast: boolean;
  /** Turn-toward-food strength: >0 steers into food, <0 away. */
  seek: number;
  /** Thrust when food is dead ahead (0..1). */
  forage: number;
  /** Baseline thrust with no food in sight (0..1). */
  cruise: number;
  /** Handedness: a persistent same-way turn (circling) when |·| is large. */
  turnBias: number;
}

/**
 * Characterise a genome by probing its policy with canonical stimuli. Returns
 * continuous traits (so even near-converged lineages read differently) plus a
 * headline label for a quick gist.
 */
export function characterizeGenome(genome: Float32Array): LineageTraits {
  const ahead = probe(genome, 1, 0, 0.8);
  const left = probe(genome, 0, 1, 0.8);
  const right = probe(genome, 0, -1, 0.8);
  const none = probe(genome, 0, 0, 0);
  const forage = ahead.thrust;
  const cruise = none.thrust;
  const seek = (left.turn - right.turn) / 2; // >0 turns toward food
  const turnBias = (left.turn + right.turn) / 2; // same-way bias -> circling

  let descKey: string;
  if (Math.abs(turnBias) > 0.35 && seek < 0.2) descKey = 'desc_circler';
  else if (seek > 0.25 && forage > 0.45) descKey = 'desc_chase';
  else if (seek > 0.25) descKey = 'desc_steer';
  else if (cruise < 0.3 && forage > 0.5) descKey = 'desc_ambush';
  else if (forage > 0.55 && cruise > 0.5) descKey = 'desc_straight';
  else if (seek < -0.2) descKey = 'desc_away';
  else descKey = 'desc_erratic';

  return { descKey, fast: (forage + cruise) / 2 > 0.55, seek, forage, cruise, turnBias };
}

export interface LineageRow {
  lineage: number;
  hue: number;
  count: number;
  trend: number;
  /** Which section to file the row under in the panel. */
  group: 'dominant' | 'distinct';
  descKey: string;
  fast: boolean;
  seek: number;
  forage: number;
  cruise: number;
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
  function rowHtml(r: LineageRow): string {
    const c = `hsl(${Math.round(r.hue * 360)}, 90%, 62%)`;
    const arrow = r.trend > 1 ? '▲' : r.trend < -1 ? '▼' : '—';
    const ac = r.trend > 1 ? '#3ff0d8' : r.trend < -1 ? '#ff5aa6' : 'rgba(207,232,255,.5)';
    const desc = `${t(r.descKey)} · ${t(r.fast ? 'fast' : 'slow')}`;
    const traits =
      `${t('tr_seek')} ${r.seek.toFixed(2)} · ` +
      `${t('tr_forage')} ${r.forage.toFixed(2)} · ` +
      `${t('tr_cruise')} ${r.cruise.toFixed(2)}`;
    return (
      `<div style="margin-bottom:9px;line-height:1.45">` +
      `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>` +
      `<b>#${r.lineage}</b> · ${r.count} <span style="color:${ac}">${arrow}</span>` +
      `<div style="opacity:.7;margin-left:16px">${desc}</div>` +
      `<div style="opacity:.5;margin-left:16px;font-size:11px">${traits}</div></div>`
    );
  }
  function sectionHtml(key: string, rows: LineageRow[]): string {
    if (rows.length === 0) return '';
    const head = `<div style="opacity:.55;letter-spacing:.08em;margin:2px 0 6px">${t(key)}</div>`;
    return head + rows.map(rowHtml).join('');
  }
  function render(): void {
    title.textContent = t('lineages');
    toggle.textContent = '🧬 ' + t('lineages');
    if (lastRows.length === 0) {
      list.innerHTML = `<div style="opacity:.5">${t('sampling')}</div>`;
      return;
    }
    list.innerHTML =
      sectionHtml(
        'dominant',
        lastRows.filter((r) => r.group === 'dominant'),
      ) +
      sectionHtml(
        'distinct',
        lastRows.filter((r) => r.group === 'distinct'),
      );
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
