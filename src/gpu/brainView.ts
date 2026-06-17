/**
 * Brain inspector: the selected creature's neural network firing in real time
 * (sensors -> hidden -> outputs, nodes coloured by activation) plus live stats.
 * Read-back layout: [0..7] inputs, [8..17] hidden, [18..19] outputs, [20] x,
 * [21] y, [22] heading, [23] speed, [24] energy, [25] hue, [26] lineage,
 * [27] alive, [28] active hidden-neuron count.
 */
import { t, onLang } from './i18n.js';
import { HIDDEN_SIZE } from '../sim/brain.js';

export interface BrainView {
  panel: HTMLElement;
  update(data: Float32Array): void;
  show(): void;
  hide(): void;
}

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
    'position:fixed;top:12px;right:12px;width:348px;padding:12px 14px;display:none;' +
    'font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;' +
    'background:rgba(2,4,10,0.72);border:1px solid rgba(63,240,216,0.22);border-radius:10px;';

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

  const stats = document.createElement('div');
  stats.style.cssText = 'margin-top:6px;line-height:1.6;';
  panel.append(header, canvas, stats);

  title.textContent = t('creatureBrain');
  track.textContent = '＋ ' + t('track');
  onLang(() => {
    title.textContent = t('creatureBrain');
    track.textContent = '＋ ' + t('track');
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
    hidden.forEach((v, i) => node(hidPos[i]!.x, hidPos[i]!.y, v, 6));
    outputs.forEach((v, i) => {
      node(outPos[i]!.x, outPos[i]!.y, v, 8);
      cx.fillStyle = 'rgba(207,232,255,0.75)';
      cx.textAlign = 'left';
      cx.fillText(t(OUTPUT_KEYS[i]!), OUT_X + 12, outPos[i]!.y);
    });
  }

  return {
    panel,
    show() {
      panel.style.display = 'block';
    },
    hide() {
      panel.style.display = 'none';
    },
    update(d) {
      const inputs = Array.from(d.subarray(0, 8));
      const hidden = Array.from(d.subarray(8, 18));
      const outputs = Array.from(d.subarray(18, 20));
      const speed = d[23]!;
      const energy = d[24]!;
      const hue = d[25]!;
      const lineage = Math.round(d[26]!);
      const alive = d[27]! >= 0.5;
      const neurons = Math.round(d[28]!);
      draw(inputs, hidden, outputs);
      const turn = outputs[0]!;
      const thrust = (outputs[1]! + 1) / 2;
      const turnTxt = turn > 0.1 ? t('turnRight') : turn < -0.1 ? t('turnLeft') : t('straight');
      const hueCss = `hsl(${Math.round(hue * 360)}, 90%, 62%)`;
      stats.innerHTML =
        `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${hueCss};margin-right:6px"></span>` +
        `${t('lineageWord')} #${lineage}${alive ? '' : ` · <span style="color:#ff5aa6">${t('deceased')}</span>`}</div>` +
        `<div>${t('energyWord')} ${energy.toFixed(1)} · ${t('speedWord')} ${speed.toFixed(1)} · ` +
        `${t('neurons')} ${neurons}/${HIDDEN_SIZE}</div>` +
        `<div>${t('decision')}: ${t('out_turn')} ${turnTxt} · ${t('out_thrust')} ${(thrust * 100).toFixed(0)}%</div>`;
    },
  };
}
