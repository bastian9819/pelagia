/**
 * Brain inspector: the selected creature's neural network firing in real time
 * (sensors -> hidden -> outputs, nodes coloured by activation) plus live stats.
 * Read-back layout: [0..16] inputs (17), [17..26] hidden, [27..29] outputs
 * (turn, thrust, attack), [30] x, [31] y, [32] heading, [33] speed, [34] energy,
 * [35] hue, [36] lineage, [37] alive, [38] active hidden-neuron count, [39] body
 * size, [40] elongation, [41] glow, [42] thermal preference, [43] toxicity.
 */
import { t, onLang } from './i18n.js';
import { icon } from './icons.js';
import { makeDraggable } from './ui.js';
import { attachTooltip } from './tooltip.js';
import { INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, WEIGHT_GENES, forward } from '../sim/brain.js';
import {
  getLineageName,
  setLineageName,
  lineageLabelHtml,
  onLineageNamesChange,
} from './lineageNames.js';

export interface BrainView {
  panel: HTMLElement;
  /** `frame` lets the decision tape advance once per sim tick (not per render). */
  update(data: Float32Array, frame: number): void;
  /** Provide the selected creature's full genome (for the static policy view). */
  setGenome(genome: Float32Array | null): void;
  show(): void;
  hide(): void;
}

// Decision tape (EEG) channels: a few sensors + the two decisions, over time.
const EEG_CHANNELS: {
  key: string;
  color: string;
  lo: number;
  hi: number;
  get: (d: Float32Array) => number;
}[] = [
  { key: 'bv_plankton', color: '#3ff0d8', lo: 0, hi: 1, get: (d) => d[2]! },
  { key: 'bv_bigfood', color: '#ffd24a', lo: 0, hi: 1, get: (d) => d[5]! },
  { key: 'bv_nbr', color: '#ff9f43', lo: 0, hi: 1, get: (d) => d[8]! },
  { key: 'energyWord', color: '#9b8cff', lo: 0, hi: 1, get: (d) => Math.min(1, d[9]!) },
  { key: 'out_turn', color: '#ff5aa6', lo: -1, hi: 1, get: (d) => d[27]! },
  { key: 'out_thrust', color: '#5ad1ff', lo: 0, hi: 1, get: (d) => (d[28]! + 1) / 2 },
  { key: 'out_attack', color: '#ff3b3b', lo: 0, hi: 1, get: (d) => Math.max(0, d[29]!) },
];
const EEG_LEN = 160; // samples kept (one per sim tick)

const INPUT_KEYS = [
  'in_planktonAhead',
  'in_planktonSide',
  'in_planktonNear',
  'in_bigfoodAhead',
  'in_bigfoodSide',
  'in_bigfoodNear',
  'in_nbrAhead',
  'in_nbrSide',
  'in_nbrNear',
  'in_energy',
  'in_speed',
  'in_temp',
  'in_school',
  'in_nbrToxin',
  'in_nbrSize',
  'in_pheroX',
  'in_pheroY',
];
const OUTPUT_KEYS = ['out_turn', 'out_thrust', 'out_attack'];

const W = 320;
const H = 230;
const IN_X = 102;
const HID_X = 188;
const OUT_X = 250;

function actColor(v: number): string {
  const m = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(63,240,216,${0.12 + 0.88 * m})` : `rgba(255,90,170,${0.12 + 0.88 * m})`;
}

export function buildBrainView(onClose: () => void, onTrack: () => void): BrainView {
  const panel = document.createElement('div');
  panel.className = 'pg-panel';
  panel.style.cssText =
    'position:fixed;top:14px;right:14px;width:348px;max-height:calc(100vh - 28px);' +
    'display:none;flex-direction:column;overflow:hidden;padding:13px 15px;font:12px var(--font-ui);z-index:10;';

  const header = document.createElement('div');
  header.style.cssText =
    'display:flex;justify-content:space-between;align-items:center;gap:8px;flex:none;margin-bottom:6px;';
  const title = document.createElement('div');
  title.className = 'pg-eyebrow';
  const right = document.createElement('div');
  right.style.cssText = 'display:flex;gap:6px;align-items:center;';
  // "Track" pins this creature into the observatory's watch-list.
  const track = document.createElement('button');
  track.className = 'pg-chip';
  let trackTimer = 0;
  const setTrack = (on: boolean): void => {
    track.innerHTML =
      icon(on ? 'check' : 'plus', 13) + `<span>${t(on ? 'tracking' : 'track')}</span>`;
  };
  track.onclick = () => {
    onTrack();
    setTrack(true);
    window.clearTimeout(trackTimer);
    trackTimer = window.setTimeout(() => setTrack(false), 1200);
  };
  const close = document.createElement('button');
  close.className = 'pg-btn pg-iconbtn';
  close.style.cssText = 'width:28px;height:28px;';
  close.innerHTML = icon('close', 16);
  close.title = t('close');
  close.onclick = onClose;
  right.append(track, close);
  header.append(title, right);

  // Editable lineage bar: the selected creature's lineage name (or #id) with a
  // pencil to christen the clade — a name layer over the functional id (P-003).
  const lineageBar = document.createElement('div');
  lineageBar.style.cssText =
    'display:flex;align-items:center;gap:7px;margin-top:8px;min-height:24px;font-size:12.5px;';
  const lDot = document.createElement('span');
  lDot.style.cssText =
    'display:inline-block;width:10px;height:10px;border-radius:50%;flex:none;background:#888;';
  const lLabel = document.createElement('span');
  lLabel.style.cssText =
    'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
  const lEdit = document.createElement('button');
  lEdit.className = 'pg-btn pg-iconbtn';
  lEdit.style.cssText = 'width:24px;height:24px;flex:none;';
  lEdit.innerHTML = icon('pencil', 13);
  lineageBar.append(lDot, lLabel, lEdit);

  // Lineage identity tracked separately from the per-frame stats so an in-progress
  // rename input is never clobbered (the bar re-renders only when it changes).
  let curLineage = -1;
  let curHue = 0;
  let curAlive = true;
  let editing = false;
  let editInput: HTMLInputElement | null = null;

  function renderLineageBar(): void {
    if (editing) return; // don't overwrite the input mid-edit
    lDot.style.background = `hsl(${Math.round(curHue * 360)}, 90%, 62%)`;
    lLabel.innerHTML =
      curLineage < 0
        ? ''
        : lineageLabelHtml(curLineage) +
          (curAlive ? '' : ` · <span style="color:#ff5aa6">${t('deceased')}</span>`);
    lEdit.style.display = curLineage < 0 ? 'none' : 'block';
  }
  function endEdit(save: boolean): void {
    const input = editInput;
    if (!input) return;
    editInput = null;
    editing = false;
    if (save) setLineageName(curLineage, input.value);
    input.remove();
    lLabel.style.display = '';
    lEdit.style.display = '';
    renderLineageBar();
  }
  function beginEdit(): void {
    if (curLineage < 0 || editing) return;
    editing = true;
    lLabel.style.display = 'none';
    lEdit.style.display = 'none';
    const input = document.createElement('input');
    input.type = 'text';
    input.value = getLineageName(curLineage) ?? '';
    input.placeholder = t('namePlaceholder');
    input.maxLength = 28;
    input.setAttribute('aria-label', t('nameLineage'));
    input.style.cssText =
      'flex:1;min-width:0;background:var(--surface-2);border:1px solid var(--border-2);' +
      'border-radius:6px;color:var(--ink);font:12.5px var(--font-ui);padding:3px 8px;outline:none;';
    // Keep keystrokes off the global shortcuts (space=pause, H=hide UI, …).
    input.onkeydown = (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') endEdit(true);
      else if (e.key === 'Escape') endEdit(false);
    };
    input.onblur = () => endEdit(true);
    editInput = input;
    lineageBar.insertBefore(input, lEdit);
    input.focus();
    input.select();
  }
  lEdit.onclick = beginEdit;
  lEdit.title = t('nameLineage');
  onLineageNamesChange(renderLineageBar); // a rename elsewhere re-labels the bar

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  canvas.style.cssText = `display:block;width:${W}px;height:${H}px;margin-top:8px;`;
  const cx = canvas.getContext('2d')!;

  // Policy view: how this brain steers vs the bearing to food / a neighbour.
  const PW = W;
  const PH = 72;
  const policyLabel = document.createElement('div');
  policyLabel.style.cssText = 'margin-top:8px;font-size:10px;opacity:.7;';
  const policy = document.createElement('canvas');
  policy.width = PW;
  policy.height = PH;
  policy.style.cssText = `display:block;width:${PW}px;height:${PH}px;`;
  const pcx = policy.getContext('2d')!;
  const listens = document.createElement('div');
  listens.style.cssText = 'margin-top:6px;font-size:11px;opacity:.75;';

  const stats = document.createElement('div');
  stats.style.cssText = 'margin-top:6px;line-height:1.6;';

  // Decision tape (EEG): sensors + decisions of this creature scrolling over time.
  const tapeLabel = document.createElement('div');
  tapeLabel.style.cssText = 'margin-top:10px;font-size:10px;opacity:.7;';
  const TW = W;
  const laneH = 22;
  const TH = EEG_CHANNELS.length * laneH;
  const tape = document.createElement('canvas');
  tape.width = TW;
  tape.height = TH;
  tape.style.cssText = `display:block;width:${TW}px;height:${TH}px;margin-top:4px;`;
  const tcx = tape.getContext('2d')!;
  const eeg: number[][] = EEG_CHANNELS.map(() => []);
  let lastEegFrame = -1;

  // Legend under the network: what the three columns and the node colours mean.
  const legend = document.createElement('div');
  legend.style.cssText =
    'margin-top:7px;display:flex;align-items:center;gap:6px 12px;flex-wrap:wrap;' +
    'font-size:10.5px;color:var(--ink-faint);';
  const dot = (rgb: string): string =>
    `<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${rgb};vertical-align:-1px;margin-right:4px"></span>`;
  function relabelLegend(): void {
    legend.innerHTML =
      `<span>${t('bv_legend_cols')}</span>` +
      `<span>${dot('rgba(63,240,216,0.95)')}${t('bv_pos')}</span>` +
      `<span>${dot('rgba(255,90,170,0.95)')}${t('bv_neg')}</span>`;
  }
  relabelLegend();

  const scroll = document.createElement('div');
  scroll.style.cssText = 'overflow:auto;flex:1;';
  scroll.append(lineageBar, canvas, legend, policyLabel, policy, listens, stats, tapeLabel, tape);
  panel.append(header, scroll);
  makeDraggable(panel, header);
  attachTooltip(title, 'panel_brain');
  attachTooltip(policyLabel, 'brain_policy');
  attachTooltip(tapeLabel, 'brain_eeg');

  // Selected creature's genome (static per creature) for the policy view.
  let genome: Float32Array | null = null;
  const sIn = new Float32Array(INPUT_SIZE);
  const sHid = new Float32Array(HIDDEN_SIZE);
  const sOut = new Float32Array(OUTPUT_SIZE);

  // `base` selects the sensor channel: 0 = plankton, 3 = big food, 6 = neighbour.
  function turnAt(base: number, bearing: number): number {
    if (!genome) return 0;
    sIn.fill(0);
    sIn[base] = Math.cos(bearing);
    sIn[base + 1] = Math.sin(bearing);
    sIn[base + 2] = 0.8;
    sIn[9] = 0.5; // energy
    sIn[10] = 0.4; // speed
    forward(genome, 0, sIn, sHid, sOut);
    return sOut[0]!;
  }

  function drawPolicy(): void {
    pcx.clearRect(0, 0, PW, PH);
    pcx.strokeStyle = 'rgba(120,160,200,0.18)';
    pcx.lineWidth = 1;
    pcx.beginPath(); // zero-turn axis
    pcx.moveTo(0, PH / 2);
    pcx.lineTo(PW, PH / 2);
    pcx.stroke();
    if (!genome) return;
    const N = 64;
    const curve = (base: number, color: string): void => {
      pcx.strokeStyle = color;
      pcx.lineWidth = 1.6;
      pcx.beginPath();
      for (let s = 0; s < N; s++) {
        const bearing = -Math.PI + (2 * Math.PI * s) / (N - 1);
        const x = (s / (N - 1)) * PW;
        const y = PH / 2 - turnAt(base, bearing) * (PH / 2 - 3);
        if (s === 0) pcx.moveTo(x, y);
        else pcx.lineTo(x, y);
      }
      pcx.stroke();
    };
    curve(0, '#3ff0d8'); // plankton response
    curve(3, '#ffd24a'); // big-food response
    curve(6, '#ff9f43'); // neighbour response
  }

  // Tally which sensor each ACTIVE hidden neuron weights most strongly.
  function neuronSummary(): string {
    if (!genome) return '';
    const groups = [0, 0, 0, 0, 0, 0, 0, 0]; // plankton,big,nbr,energy,speed,temp,school,phero
    for (let h = 0; h < HIDDEN_SIZE; h++) {
      if (genome[WEIGHT_GENES + h]! < 0) continue; // disabled
      const base = h * (INPUT_SIZE + 1);
      let bestI = 0;
      let bestW = -1;
      for (let i = 0; i < INPUT_SIZE; i++) {
        const w = Math.abs(genome[base + i]!);
        if (w > bestW) {
          bestW = w;
          bestI = i;
        }
      }
      // prettier-ignore
      // 13/14 (neighbour toxicity/size) fold into the neighbour group (2); 15/16 = pheromone (7).
      const g = bestI < 3 ? 0 : bestI < 6 ? 1 : bestI < 9 ? 2 : bestI === 9 ? 3
              : bestI === 10 ? 4 : bestI === 11 ? 5 : bestI === 12 ? 6 : bestI < 15 ? 2 : 7;
      groups[g]!++;
    }
    const parts: string[] = [];
    const labels = [
      t('bv_plankton'),
      t('bv_bigfood'),
      t('bv_nbr'),
      t('energyWord'),
      t('speedWord'),
      t('tempPref'),
      t('bv_school'),
      t('bv_phero'),
    ];
    for (let g = 0; g < labels.length; g++)
      if (groups[g]! > 0) parts.push(`${labels[g]}×${groups[g]}`);
    return `${t('bv_listens')}: ${parts.join(' · ')}`;
  }

  function refreshPolicy(): void {
    policyLabel.textContent = `${t('bv_policy')} — ${t('bv_plankton')} ▬ · ${t('bv_bigfood')} ▬ · ${t('bv_nbr')} ▬`;
    drawPolicy();
    listens.textContent = neuronSummary();
  }

  function clearTape(): void {
    for (const c of eeg) c.length = 0;
    lastEegFrame = -1;
    drawTape();
  }

  // One sample per sim tick (deduped by frame, so a paused tape freezes and the
  // step button advances it exactly one decision at a time).
  function pushTape(d: Float32Array, frame: number): void {
    if (frame === lastEegFrame) return;
    lastEegFrame = frame;
    for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
      const arr = eeg[ch]!;
      arr.push(EEG_CHANNELS[ch]!.get(d));
      if (arr.length > EEG_LEN) arr.shift();
    }
    drawTape();
  }

  function drawTape(): void {
    tcx.clearRect(0, 0, TW, TH);
    for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
      const def = EEG_CHANNELS[ch]!;
      const top = ch * laneH;
      // lane separator + baseline
      tcx.strokeStyle = 'rgba(120,160,200,0.10)';
      tcx.lineWidth = 1;
      tcx.beginPath();
      tcx.moveTo(0, top + laneH - 0.5);
      tcx.lineTo(TW, top + laneH - 0.5);
      tcx.stroke();
      // label
      tcx.fillStyle = 'rgba(207,232,255,0.55)';
      tcx.font = '9px ui-monospace, monospace';
      tcx.textBaseline = 'top';
      tcx.textAlign = 'left';
      tcx.fillText(t(def.key), 3, top + 2);
      // signal
      const arr = eeg[ch]!;
      if (arr.length >= 2) {
        tcx.strokeStyle = def.color;
        tcx.lineWidth = 1.4;
        tcx.beginPath();
        for (let i = 0; i < arr.length; i++) {
          const x = (i / (EEG_LEN - 1)) * TW;
          const norm = (arr[i]! - def.lo) / (def.hi - def.lo);
          const y = top + (laneH - 3) - Math.max(0, Math.min(1, norm)) * (laneH - 5);
          if (i === 0) tcx.moveTo(x, y);
          else tcx.lineTo(x, y);
        }
        tcx.stroke();
      }
    }
  }

  title.textContent = t('creatureBrain');
  setTrack(false);
  tapeLabel.textContent = t('eeg_title');
  onLang(() => {
    title.textContent = t('creatureBrain');
    setTrack(false);
    close.title = t('close');
    lEdit.title = t('nameLineage');
    tapeLabel.textContent = t('eeg_title');
    relabelLegend();
    renderLineageBar();
    refreshPolicy();
    drawTape();
  });

  const colY = (count: number, i: number, top: number, bottom: number): number =>
    count === 1 ? (top + bottom) / 2 : top + ((bottom - top) * i) / (count - 1);

  function draw(inputs: number[], hidden: number[], outputs: number[]): void {
    cx.clearRect(0, 0, W, H);
    const inPos = inputs.map((_, i) => ({ x: IN_X, y: colY(inputs.length, i, 18, H - 12) }));
    const hidPos = hidden.map((_, i) => ({ x: HID_X, y: colY(10, i, 12, H - 12) }));
    const outPos = outputs.map((_, i) => ({
      x: OUT_X,
      y: colY(outputs.length, i, H * 0.32, H * 0.68),
    }));

    cx.strokeStyle = 'rgba(120,160,200,0.06)';
    cx.lineWidth = 1;
    cx.beginPath();
    for (const a of inPos)
      for (const b of hidPos) {
        cx.moveTo(a.x, a.y);
        cx.lineTo(b.x, b.y);
      }
    for (const a of hidPos)
      for (const b of outPos) {
        cx.moveTo(a.x, a.y);
        cx.lineTo(b.x, b.y);
      }
    cx.stroke();

    const node = (x: number, y: number, v: number, r: number): void => {
      cx.beginPath();
      cx.arc(x, y, r, 0, Math.PI * 2);
      cx.fillStyle = actColor(v);
      cx.fill();
      cx.strokeStyle = 'rgba(207,232,255,0.25)';
      cx.lineWidth = 1;
      cx.stroke();
    };
    cx.font = '9px ui-monospace, monospace';
    cx.textBaseline = 'middle';
    inputs.forEach((v, i) => {
      node(inPos[i]!.x, inPos[i]!.y, v, 6);
      cx.fillStyle = 'rgba(207,232,255,0.6)';
      cx.textAlign = 'right';
      cx.fillText(t(INPUT_KEYS[i]!), IN_X - 10, inPos[i]!.y);
    });
    hidden.forEach((v, i) => {
      const disabled = genome ? genome[WEIGHT_GENES + i]! < 0 : false;
      if (disabled) {
        // Switched-off neuron: a hollow ring, so evolved complexity is visible.
        cx.beginPath();
        cx.arc(hidPos[i]!.x, hidPos[i]!.y, 5, 0, Math.PI * 2);
        cx.strokeStyle = 'rgba(120,160,200,0.4)';
        cx.lineWidth = 1;
        cx.stroke();
      } else {
        node(hidPos[i]!.x, hidPos[i]!.y, v, 6);
      }
    });
    outputs.forEach((v, i) => {
      node(outPos[i]!.x, outPos[i]!.y, v, 8);
      cx.fillStyle = 'rgba(207,232,255,0.75)';
      cx.textAlign = 'left';
      cx.fillText(t(OUTPUT_KEYS[i]!), OUT_X + 12, outPos[i]!.y);
    });
  }

  return {
    panel,
    setGenome(g) {
      genome = g;
      endEdit(false); // selection changed -> drop any in-progress rename
      if (g === null) {
        curLineage = -1;
        renderLineageBar();
      }
      clearTape(); // new selection -> fresh tape
      refreshPolicy();
    },
    show() {
      panel.style.display = 'flex';
    },
    hide() {
      panel.style.display = 'none';
    },
    update(d, frame) {
      pushTape(d, frame);
      const inputs = Array.from(d.subarray(0, 17));
      const hidden = Array.from(d.subarray(17, 27));
      const outputs = Array.from(d.subarray(27, 30));
      const speed = d[33]!;
      const energy = d[34]!;
      const hue = d[35]!;
      const lineage = Math.round(d[36]!);
      const alive = d[37]! >= 0.5;
      if (lineage !== curLineage || hue !== curHue || alive !== curAlive) {
        curLineage = lineage;
        curHue = hue;
        curAlive = alive;
        renderLineageBar();
      }
      const neurons = Math.round(d[38]!);
      const bodySize = d[39]!;
      const elong = d[40]!;
      const glow = d[41]!;
      const thermal = d[42]!;
      const toxin = d[43]!;
      const shapeTxt =
        elong > 1.15 ? t('shapeEel') : elong < 0.85 ? t('shapeBlob') : t('shapeOval');
      const thermalTxt =
        thermal > 0.25 ? t('tempWarm') : thermal < -0.25 ? t('tempCold') : t('tempMild');
      draw(inputs, hidden, outputs);
      const turn = outputs[0]!;
      const thrust = (outputs[1]! + 1) / 2;
      const attack = outputs[2]!;
      const turnTxt = turn > 0.1 ? t('turnRight') : turn < -0.1 ? t('turnLeft') : t('straight');
      // Lineage identity (name/#id + deceased) now lives in the editable lineageBar.
      stats.innerHTML =
        `<div>${t('energyWord')} ${energy.toFixed(1)} · ${t('speedWord')} ${speed.toFixed(1)} · ` +
        `${t('sizeWord')} ${bodySize.toFixed(2)}×</div>` +
        `<div>${t('shapeWord')} ${shapeTxt} · ${t('glowWord')} ${glow.toFixed(2)}× · ${t('tempPref')} ${thermalTxt}` +
        `${toxin > 0.05 ? ` · <span style="color:#bfff3a">${t('toxinWord')} ${(toxin * 100).toFixed(0)}%</span>` : ''}</div>` +
        `<div>${t('neurons')} ${neurons}/${HIDDEN_SIZE}</div>` +
        `<div>${t('decision')}: ${t('out_turn')} ${turnTxt} · ${t('out_thrust')} ${(thrust * 100).toFixed(0)}%` +
        `${attack > 0 ? ` · <span style="color:#ff3b3b">${t('out_attack')} ▲</span>` : ''}</div>`;
    },
  };
}
