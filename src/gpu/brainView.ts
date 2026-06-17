/**
 * The brain inspector panel: draws the selected creature's neural network firing
 * in real time (input sensors -> hidden -> outputs, nodes coloured by
 * activation) plus its live stats. Fed by the inspect read-back buffer.
 *
 * Read-back layout (floats): [0..7] inputs, [8..17] hidden, [18..19] outputs,
 * [20] x, [21] y, [22] heading, [23] speed, [24] energy, [25] hue, [26] age,
 * [27] alive.
 */
export interface BrainView {
  panel: HTMLElement;
  update(data: Float32Array): void;
  show(): void;
  hide(): void;
}

const INPUT_LABELS = [
  'food ahead',
  'food side',
  'food near',
  'nbr ahead',
  'nbr side',
  'nbr near',
  'energy',
  'speed',
];
const OUTPUT_LABELS = ['turn', 'thrust'];

function actColor(v: number): string {
  const m = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(63,240,216,${0.12 + 0.88 * m})` : `rgba(255,90,170,${0.12 + 0.88 * m})`;
}

export function buildBrainView(onClose: () => void): BrainView {
  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;top:12px;right:12px;width:300px;padding:12px 14px;display:none;' +
    'font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;' +
    'background:rgba(2,4,10,0.72);border:1px solid rgba(63,240,216,0.22);border-radius:10px;';

  const header = document.createElement('div');
  header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;';
  const title = document.createElement('div');
  title.textContent = 'creature brain';
  title.style.cssText = 'font-weight:600;letter-spacing:.1em;color:#3ff0d8;';
  const close = document.createElement('button');
  close.textContent = '×';
  close.style.cssText =
    'background:none;border:none;color:#cfe8ff;font-size:18px;cursor:pointer;line-height:1;';
  close.onclick = onClose;
  header.append(title, close);

  const canvas = document.createElement('canvas');
  canvas.width = 300;
  canvas.height = 230;
  canvas.style.cssText = 'display:block;width:300px;height:230px;margin-top:8px;';
  const cx = canvas.getContext('2d')!;

  const stats = document.createElement('div');
  stats.style.cssText = 'margin-top:6px;line-height:1.6;';

  panel.append(header, canvas, stats);

  // Node positions.
  const inX = 60;
  const hidX = 150;
  const outX = 245;
  const colY = (count: number, i: number, top: number, bottom: number): number =>
    count === 1 ? (top + bottom) / 2 : top + ((bottom - top) * i) / (count - 1);

  function draw(inputs: number[], hidden: number[], outputs: number[]): void {
    const w = canvas.width;
    const h = canvas.height;
    cx.clearRect(0, 0, w, h);
    const inPos = inputs.map((_, i) => ({ x: inX, y: colY(8, i, 18, h - 12) }));
    const hidPos = hidden.map((_, i) => ({ x: hidX, y: colY(10, i, 12, h - 12) }));
    const outPos = outputs.map((_, i) => ({ x: outX, y: colY(2, i, h * 0.35, h * 0.65) }));

    // Faint topology edges.
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
    cx.font = '10px ui-monospace, monospace';
    cx.textBaseline = 'middle';
    inputs.forEach((v, i) => {
      node(inPos[i]!.x, inPos[i]!.y, v, 6);
      cx.fillStyle = 'rgba(207,232,255,0.55)';
      cx.textAlign = 'right';
      cx.fillText(INPUT_LABELS[i]!, inX - 10, inPos[i]!.y);
    });
    hidden.forEach((v, i) => node(hidPos[i]!.x, hidPos[i]!.y, v, 6));
    outputs.forEach((v, i) => {
      node(outPos[i]!.x, outPos[i]!.y, v, 8);
      cx.fillStyle = 'rgba(207,232,255,0.7)';
      cx.textAlign = 'left';
      cx.fillText(OUTPUT_LABELS[i]!, outX + 12, outPos[i]!.y);
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
      draw(inputs, hidden, outputs);
      const turn = outputs[0]!;
      const thrust = (outputs[1]! + 1) / 2;
      const hueCss = `hsl(${Math.round(hue * 360)}, 90%, 62%)`;
      stats.innerHTML =
        `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${hueCss};margin-right:6px"></span>` +
        `lineage #${lineage}${alive ? '' : ' · <span style="color:#ff5aa6">deceased</span>'}</div>` +
        `<div>energy ${energy.toFixed(1)} · speed ${speed.toFixed(1)}</div>` +
        `<div>decision: turn ${turn > 0.1 ? 'right ▶' : turn < -0.1 ? '◀ left' : '— straight'} · ` +
        `thrust ${(thrust * 100).toFixed(0)}%</div>`;
    },
  };
}
