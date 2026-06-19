/**
 * Lineage explorer: ranks dominant lineages and characterises each one's neural
 * policy by probing a representative brain with canonical stimuli (food ahead /
 * to the side / absent) using the CPU forward pass. Answers "what neural
 * decisions made this lineage dominate?". Foraging only — aggression/predation
 * arrives in Phase 6 (creatures can't eat each other yet).
 */
import {
  forward,
  INPUT_SIZE,
  HIDDEN_SIZE,
  OUTPUT_SIZE,
  WEIGHT_GENES,
  SIZE_GENE,
  sizeFromGene,
} from '../sim/brain.js';
import { t, onLang } from './i18n.js';
import { icon } from './icons.js';

const inp = new Float32Array(INPUT_SIZE);
const hid = new Float32Array(HIDDEN_SIZE);
const out = new Float32Array(OUTPUT_SIZE);

function probe(
  genome: Float32Array,
  planktonCos: number,
  planktonSin: number,
  planktonProx: number,
  bigCos = 0,
  bigSin = 0,
  bigProx = 0,
  nbrCos = 0,
  nbrSin = 0,
  nbrProx = 0,
): { turn: number; thrust: number; attack: number } {
  inp.fill(0);
  inp[0] = planktonCos;
  inp[1] = planktonSin;
  inp[2] = planktonProx;
  inp[3] = bigCos;
  inp[4] = bigSin;
  inp[5] = bigProx;
  inp[6] = nbrCos;
  inp[7] = nbrSin;
  inp[8] = nbrProx;
  inp[9] = 0.5;
  inp[10] = 0.4;
  forward(genome, 0, inp, hid, out);
  return { turn: out[0]!, thrust: (out[1]! + 1) / 2, attack: out[2]! };
}

/** Continuous behavioural traits of a genome + a headline label. */
export interface LineageTraits {
  descKey: string;
  fast: boolean;
  /** Turn-toward-plankton strength: >0 steers into plankton, <0 away. */
  seek: number;
  /** Phase 6: turn-toward-big-food strength. >0 hunts big-food blooms. */
  bigSeek: number;
  /** Thrust when food is dead ahead (0..1). */
  forage: number;
  /** Baseline thrust with no food in sight (0..1). */
  cruise: number;
  /** Handedness: a persistent same-way turn (circling) when |·| is large. */
  turnBias: number;
  /** Phase 6: steer-toward-neighbour strength. >0 hunts, <0 flees others. */
  aggression: number;
  /** Attack intent (out[2]) when a neighbour is near — >0 means it chooses to prey. */
  attackDrive: number;
  /** Phase 6: how many hidden neurons this brain has switched on (0..HIDDEN_SIZE). */
  neurons: number;
  /** Phase 6: evolved body-size multiplier. */
  size: number;
}

/**
 * Characterise a genome by probing its policy with canonical stimuli. Returns
 * continuous traits (so even near-converged lineages read differently) plus a
 * headline label for a quick gist. Phase 6 also probes the neighbour sensors to
 * read aggression (does the brain steer toward another creature?).
 */
export function characterizeGenome(genome: Float32Array): LineageTraits {
  const ahead = probe(genome, 1, 0, 0.8);
  const left = probe(genome, 0, 1, 0.8);
  const right = probe(genome, 0, -1, 0.8);
  const none = probe(genome, 0, 0, 0);
  const bigLeft = probe(genome, 0, 0, 0, 0, 1, 0.8);
  const bigRight = probe(genome, 0, 0, 0, 0, -1, 0.8);
  const nbrLeft = probe(genome, 0, 0, 0, 0, 0, 0, 0, 1, 0.8);
  const nbrRight = probe(genome, 0, 0, 0, 0, 0, 0, 0, -1, 0.8);
  const forage = ahead.thrust;
  const cruise = none.thrust;
  const seek = (left.turn - right.turn) / 2; // >0 turns toward plankton
  const bigSeek = (bigLeft.turn - bigRight.turn) / 2; // >0 turns toward big food
  const turnBias = (left.turn + right.turn) / 2; // same-way bias -> circling
  const aggression = (nbrLeft.turn - nbrRight.turn) / 2; // >0 turns toward neighbour
  const attackDrive = (nbrLeft.attack + nbrRight.attack) / 2; // >0 chooses to attack a neighbour
  let neurons = 0;
  for (let h = 0; h < HIDDEN_SIZE; h++) if (genome[WEIGHT_GENES + h]! >= 0) neurons++;
  const size = sizeFromGene(genome[SIZE_GENE]!);

  let descKey: string;
  // A real predator now both turns toward AND chooses to attack a neighbour.
  if (attackDrive > 0.15 && aggression > 0.05) descKey = 'desc_predator';
  else if (aggression < -0.35) descKey = 'desc_skittish';
  else if (bigSeek > 0.3 && bigSeek > seek + 0.12) descKey = 'desc_bigfeeder';
  else if (Math.abs(turnBias) > 0.35 && seek < 0.2) descKey = 'desc_circler';
  else if (seek > 0.25 && forage > 0.45) descKey = 'desc_chase';
  else if (seek > 0.25) descKey = 'desc_steer';
  else if (cruise < 0.3 && forage > 0.5) descKey = 'desc_ambush';
  else if (forage > 0.55 && cruise > 0.5) descKey = 'desc_straight';
  else if (seek < -0.2) descKey = 'desc_away';
  else descKey = 'desc_erratic';

  return {
    descKey,
    fast: (forage + cruise) / 2 > 0.55,
    seek,
    bigSeek,
    forage,
    cruise,
    turnBias,
    aggression,
    attackDrive,
    neurons,
    size,
  };
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
  bigSeek: number;
  forage: number;
  cruise: number;
  aggression: number;
  neurons: number;
}

export interface LineagePanel {
  panel: HTMLElement;
  toggle: HTMLButtonElement;
  update(rows: LineageRow[]): void;
}

export function buildLineagePanel(): LineagePanel {
  const panel = document.createElement('div');
  panel.className = 'pg-panel';
  panel.style.cssText =
    'position:fixed;left:14px;top:236px;width:266px;max-height:calc(100vh - 320px);overflow:auto;' +
    'display:none;padding:14px 15px;z-index:7;';
  const title = document.createElement('div');
  title.className = 'pg-eyebrow';
  title.style.cssText = 'margin-bottom:10px;';
  const list = document.createElement('div');
  list.style.cssText = 'font:12px var(--font-ui);';
  panel.append(title, list);

  const toggle = document.createElement('button');
  toggle.className = 'pg-btn';
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
      `${t('tr_big')} ${r.bigSeek.toFixed(2)} · ` +
      `${t('tr_aggr')} ${r.aggression.toFixed(2)} · ` +
      `${r.neurons} ${t('tr_neurons')}`;
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
    const head = `<div class="pg-eyebrow" style="margin:12px 0 7px">${t(key)}</div>`;
    return head + rows.map(rowHtml).join('');
  }
  function render(): void {
    title.textContent = t('lineages');
    toggle.innerHTML = icon('list', 16) + `<span>${t('lineages')}</span>`;
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
