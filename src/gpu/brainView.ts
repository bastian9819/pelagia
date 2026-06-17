/**
 * Brain inspector: the selected creature's neural network firing in real time
 * (sensors -> hidden -> outputs, nodes coloured by activation) plus live stats.
 * Read-back layout: [0..7] inputs, [8..17] hidden, [18..19] outputs, [20] x,
 * [21] y, [22] heading, [23] speed, [24] energy, [25] hue, [26] lineage,
 * [27] alive, [28] active hidden-neuron count, [29] body size.
 */
import { t, onLang } from './i18n.js';
import { INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, WEIGHT_GENES, forward } from '../sim/brain.js';

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
  { key: 'bv_food', color: '#3ff0d8', lo: 0, hi: 1, get: (d) => d[2]! },
  { key: 'bv_nbr', color: '#ff9f43', lo: 0, hi: 1, get: (d) => d[5]! },
  { key: 'energyWord', color: '#9b8cff', lo: 0, hi: 1, get: (d) => Math.min(1, d[6]!) },
  { key: 'out_turn', color: '#ff5aa6', lo: -1, hi: 1, get: (d) => d[18]! },
  { key: 'out_thrust', color: '#5ad1ff', lo: 0, hi: 1, get: (d) => (d[19]! + 1) / 2 },
];
const EEG_LEN = 160; // samples kept (one per sim tick)

const INPUT_KEYS = [
  'in_foodAhead',
  'in_foodSide',
  'in_foodNear',
  'in_nbrAhead',
  'in_nbrSide',
  'in_nbrNear',
  'in_energy',
  'in_speed',
];
const OUTPUT_KEYS = ['out_turn', 'out_thrust'];

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
  panel.style.cssText =
    'position:fixed;top:12px;right:12px;width:348px;max-height:calc(100vh - 24px);overflow:auto;' +
    'padding:12px 14px;display:none;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;' +
    'color:#cfe8ff;background:rgba(2,4,10,0.72);border:1px solid rgba(63,240,216,0.22);border-radius:10px;';

  const header = document.createElement('div');
  header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;';
  const title = document.createElement('div');
  title.style.cssText = 'font-weight:600;letter-spacing:.1em;color:#3ff0d8;';
  const right = document.createElement('div');
  right.style.cssText = 'display:flex;gap:8px;align-items:center;';
  // "Track" pins this creature into the observatory's watch-list.
  const track = document.createElement('button');
  track.style.cssText =
    'padding:3px 9px;background:rgba(63,240,216,0.14);color:#cfe8ff;border:1px solid ' +
    'rgba(63,240,216,0.3);border-radius:7px;cursor:pointer;font:inherit;font-size:11px;';
  let trackTimer = 0;
  track.onclick = () => {
    onTrack();
    track.textContent = '✓ ' + t('tracking');
    window.clearTimeout(trackTimer);
    trackTimer = window.setTimeout(() => (track.textContent = '＋ ' + t('track')), 1200);
  };
  const close = document.createElement('button');
  close.textContent = '×';
  close.style.cssText =
    'background:none;border:none;color:#cfe8ff;font-size:18px;cursor:pointer;line-height:1;';
  close.onclick = onClose;
  right.append(track, close);
  header.append(title, right);

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

  panel.append(header, canvas, policyLabel, policy, listens, stats, tapeLabel, tape);

  // Selected creature's genome (static per creature) for the policy view.
  let genome: Float32Array | null = null;
  const sIn = new Float32Array(INPUT_SIZE);
  const sHid = new Float32Array(HIDDEN_SIZE);
  const sOut = new Float32Array(OUTPUT_SIZE);

  function turnAt(food: boolean, bearing: number): number {
    if (!genome) return 0;
    sIn.fill(0);
    sIn[food ? 0 : 3] = Math.cos(bearing);
    sIn[food ? 1 : 4] = Math.sin(bearing);
    sIn[food ? 2 : 5] = 0.8;
    sIn[6] = 0.5;
    sIn[7] = 0.4;
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
    const curve = (food: boolean, color: string): void => {
      pcx.strokeStyle = color;
      pcx.lineWidth = 1.6;
      pcx.beginPath();
      for (let s = 0; s < N; s++) {
        const bearing = -Math.PI + (2 * Math.PI * s) / (N - 1);
        const x = (s / (N - 1)) * PW;
        const y = PH / 2 - turnAt(food, bearing) * (PH / 2 - 3);
        if (s === 0) pcx.moveTo(x, y);
        else pcx.lineTo(x, y);
      }
      pcx.stroke();
    };
    curve(true, '#3ff0d8'); // food response
    curve(false, '#ff9f43'); // neighbour response
  }

  // Tally which sensor each ACTIVE hidden neuron weights most strongly.
  function neuronSummary(): string {
    if (!genome) return '';
    const groups = [0, 0, 0, 0]; // food, neighbour, energy, speed
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
      const g = bestI < 3 ? 0 : bestI < 6 ? 1 : bestI === 6 ? 2 : 3;
      groups[g]!++;
    }
    const parts: string[] = [];
    const labels = [t('bv_food'), t('bv_nbr'), t('energyWord'), t('speedWord')];
    for (let g = 0; g < 4; g++) if (groups[g]! > 0) parts.push(`${labels[g]}×${groups[g]}`);
    return `${t('bv_listens')}: ${parts.join(' · ')}`;
  }

  function refreshPolicy(): void {
    policyLabel.textContent = `${t('bv_policy')} — ${t('bv_food')} ▬ · ${t('bv_nbr')} ▬`;
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
  track.textContent = '＋ ' + t('track');
  tapeLabel.textContent = t('eeg_title');
  onLang(() => {
    title.textContent = t('creatureBrain');
    track.textContent = '＋ ' + t('track');
    tapeLabel.textContent = t('eeg_title');
    refreshPolicy();
    drawTape();
  });

  const colY = (count: number, i: number, top: number, bottom: number): number =>
    count === 1 ? (top + bottom) / 2 : top + ((bottom - top) * i) / (count - 1);

  function draw(inputs: number[], hidden: number[], outputs: number[]): void {
    cx.clearRect(0, 0, W, H);
    const inPos = inputs.map((_, i) => ({ x: IN_X, y: colY(8, i, 18, H - 12) }));
    const hidPos = hidden.map((_, i) => ({ x: HID_X, y: colY(10, i, 12, H - 12) }));
    const outPos = outputs.map((_, i) => ({ x: OUT_X, y: colY(2, i, H * 0.35, H * 0.65) }));

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
      clearTape(); // new selection -> fresh tape
      refreshPolicy();
    },
    show() {
      panel.style.display = 'block';
    },
    hide() {
      panel.style.display = 'none';
    },
    update(d, frame) {
      pushTape(d, frame);
      const inputs = Array.from(d.subarray(0, 8));
      const hidden = Array.from(d.subarray(8, 18));
      const outputs = Array.from(d.subarray(18, 20));
      const speed = d[23]!;
      const energy = d[24]!;
      const hue = d[25]!;
      const lineage = Math.round(d[26]!);
      const alive = d[27]! >= 0.5;
      const neurons = Math.round(d[28]!);
      const bodySize = d[29]!;
      draw(inputs, hidden, outputs);
      const turn = outputs[0]!;
      const thrust = (outputs[1]! + 1) / 2;
      const turnTxt = turn > 0.1 ? t('turnRight') : turn < -0.1 ? t('turnLeft') : t('straight');
      const hueCss = `hsl(${Math.round(hue * 360)}, 90%, 62%)`;
      stats.innerHTML =
        `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${hueCss};margin-right:6px"></span>` +
        `${t('lineageWord')} #${lineage}${alive ? '' : ` · <span style="color:#ff5aa6">${t('deceased')}</span>`}</div>` +
        `<div>${t('energyWord')} ${energy.toFixed(1)} · ${t('speedWord')} ${speed.toFixed(1)} · ` +
        `${t('sizeWord')} ${bodySize.toFixed(2)}×</div>` +
        `<div>${t('neurons')} ${neurons}/${HIDDEN_SIZE}</div>` +
        `<div>${t('decision')}: ${t('out_turn')} ${turnTxt} · ${t('out_thrust')} ${(thrust * 100).toFixed(0)}%</div>`;
    },
  };
}
