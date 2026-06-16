import './style.css';

/**
 * Phase 0.0 entry point.
 *
 * Nothing is alive yet — this just proves the toolchain is wired up and paints
 * the bioluminescent deep-ocean backdrop that the living simulation will sit on
 * top of in later phases.
 */

const canvas = document.querySelector<HTMLCanvasElement>('#ocean');
if (!canvas) throw new Error('Canvas #ocean not found');

const ctx = canvas.getContext('2d');
if (!ctx) throw new Error('2D context unavailable');

function resize(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function paint(ctx: CanvasRenderingContext2D): void {
  const w = window.innerWidth;
  const h = window.innerHeight;

  // Deep-ocean radial gradient: a faint glow at the centre fading to the abyss.
  const cx = w / 2;
  const cy = h / 2;
  const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.75);
  gradient.addColorStop(0, '#0b1f3a');
  gradient.addColorStop(0.55, '#061229');
  gradient.addColorStop(1, '#02040a');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);

  // Wordmark.
  ctx.textAlign = 'center';
  ctx.fillStyle = '#3ff0d8';
  ctx.font = '600 clamp(2rem, 9vw, 6rem) ui-sans-serif, system-ui, sans-serif';
  ctx.fillText('PELAGIA', cx, cy);

  ctx.fillStyle = 'rgba(207, 232, 255, 0.55)';
  ctx.font = '400 clamp(0.7rem, 2.2vw, 1rem) ui-sans-serif, system-ui, sans-serif';
  ctx.fillText('an ocean of artificial life — phase 0', cx, cy + 48);
}

function render(): void {
  if (!canvas || !ctx) return;
  resize(canvas, ctx);
  paint(ctx);
}

render();
window.addEventListener('resize', render);
