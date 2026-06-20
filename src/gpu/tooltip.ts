/**
 * A small custom tooltip system. Native `title` tooltips are slow to appear and
 * barely noticeable, so the UI explains itself with styled, two-level tooltips:
 * a friendly plain-language line (for non-technical visitors) plus an optional
 * technical line (range, formula, units) for the curious. Bilingual via getLang().
 *
 * One shared tooltip element is positioned next to whatever you hover. Attach with
 * `attachTooltip(el, key)`; content lives in TIPS keyed by a short id.
 */
import { getLang } from './i18n.js';

interface Tip {
  title: string;
  body: string;
  tech?: string;
}

type TipEntry = { en: Tip; es: Tip };

// Tooltip copy. Friendly first, technical second. Keep `body` jargon-free.
const TIPS: Record<string, TipEntry> = {
  // --- Transport bar ---
  pause: {
    en: { title: 'Pause / play', body: 'Freeze the ocean or let it run.', tech: 'Shortcut: space' },
    es: {
      title: 'Pausa / play',
      body: 'Congela el océano o déjalo correr.',
      tech: 'Atajo: espacio',
    },
  },
  step: {
    en: {
      title: 'Step one tick',
      body: 'Advance the simulation by a single tick to study one decision frame by frame.',
      tech: 'Pauses first, then advances exactly one tick',
    },
    es: {
      title: 'Avanzar un tick',
      body: 'Adelanta la simulación un solo tick para estudiar una decisión fotograma a fotograma.',
      tech: 'Primero pausa, luego avanza exactamente un tick',
    },
  },
  speed: {
    en: {
      title: 'Speed',
      body: 'How fast time flows. Slow motion to study behaviour, turbo to fast-forward evolution.',
      tech: '0.1× (slow-mo) … 1× (real time) … 16× (turbo) ticks per frame',
    },
    es: {
      title: 'Velocidad',
      body: 'A qué ritmo fluye el tiempo. Cámara lenta para estudiar, turbo para adelantar la evolución.',
      tech: '0.1× (lenta) … 1× (tiempo real) … 16× (turbo) ticks por fotograma',
    },
  },
  fit: {
    en: { title: 'Fit to view', body: 'Reset the camera to frame the whole ocean.' },
    es: { title: 'Encajar la vista', body: 'Reinicia la cámara para ver el océano entero.' },
  },
  color: {
    en: {
      title: 'Colour by trait',
      body: 'Paint every creature by a chosen trait so you can watch it spread across the ocean as it evolves. Cycle through lineage, size, neurons, energy and more.',
      tech: 'A blue→red ramp = low→high; a legend appears next to the bar',
    },
    es: {
      title: 'Colorear por rasgo',
      body: 'Pinta cada criatura según un rasgo elegido para ver cómo se extiende por el océano al evolucionar. Cicla entre linaje, tamaño, neuronas, energía y más.',
      tech: 'Rampa azul→rojo = bajo→alto; aparece una leyenda junto a la barra',
    },
  },
  menu: {
    en: {
      title: 'Menu',
      body: 'Panels and tools: lineages, god mode, observatory, history and more.',
    },
    es: {
      title: 'Menú',
      body: 'Paneles y herramientas: linajes, modo dios, observatorio, historia y más.',
    },
  },
  // --- HUD ---
  alive: {
    en: {
      title: 'Creatures alive',
      body: 'How many creatures are alive right now. The little graph shows it rising and falling over time.',
      tech: 'Carrying capacity is set by food, metabolism and predation',
    },
    es: {
      title: 'Criaturas vivas',
      body: 'Cuántas criaturas hay vivas ahora mismo. La gráfica muestra cómo sube y baja con el tiempo.',
      tech: 'La capacidad de carga la fijan la comida, el metabolismo y la depredación',
    },
  },
  tick: {
    en: {
      title: 'Tick',
      body: 'One heartbeat of the simulation: in each tick every creature senses, decides and moves once. The counter shows how many have passed.',
      tech: '≈ 60 ticks per second at 1× speed',
    },
    es: {
      title: 'Tick',
      body: 'Un latido de la simulación: en cada tick toda criatura percibe, decide y se mueve una vez. El contador muestra cuántos han pasado.',
      tech: '≈ 60 ticks por segundo a velocidad 1×',
    },
  },
  // --- Brush dock ---
  tool_pan: {
    en: {
      title: 'Move / select',
      body: 'Drag to pan the camera; click a creature to open its brain. The default tool.',
    },
    es: {
      title: 'Mover / seleccionar',
      body: 'Arrastra para mover la cámara; haz clic en una criatura para ver su cerebro. La herramienta por defecto.',
    },
  },
  tool_attract: {
    en: {
      title: 'Magnet',
      body: 'Drag over the ocean to pull nearby creatures toward your cursor. The dashed ring shows the area it affects.',
      tech: 'Force grows toward the centre of the brush',
    },
    es: {
      title: 'Imán',
      body: 'Arrastra sobre el océano para atraer criaturas hacia el cursor. El anillo discontinuo marca la zona afectada.',
      tech: 'La fuerza crece hacia el centro del pincel',
    },
  },
  tool_repel: {
    en: {
      title: 'Repel',
      body: 'Drag to push creatures away from your cursor, clearing a space in the crowd.',
      tech: 'Same force as the magnet, reversed',
    },
    es: {
      title: 'Espantar',
      body: 'Arrastra para alejar criaturas del cursor, despejando un hueco entre la multitud.',
      tech: 'La misma fuerza que el imán, invertida',
    },
  },
  tool_food: {
    en: {
      title: 'Feed',
      body: 'Drag to scatter food where you paint, drawing creatures in to graze.',
    },
    es: {
      title: 'Alimentar',
      body: 'Arrastra para esparcir comida donde pintas, atrayendo criaturas a comer.',
    },
  },
  tool_heal: {
    en: {
      title: 'Heal',
      body: 'Drag to give energy to the creatures under the brush, keeping them alive longer.',
    },
    es: {
      title: 'Curar',
      body: 'Arrastra para dar energía a las criaturas bajo el pincel, manteniéndolas vivas más tiempo.',
    },
  },
  tool_seed: {
    en: {
      title: 'Seed',
      body: 'Drop new creatures where you paint. If one is selected it clones its brain (spreading its lineage); otherwise it makes a brand-new random lineage.',
    },
    es: {
      title: 'Sembrar',
      body: 'Suelta criaturas nuevas donde pintas. Si hay una seleccionada, clona su cerebro (extiende su linaje); si no, crea un linaje nuevo aleatorio.',
    },
  },
  tool_mutagen: {
    en: {
      title: 'Mutagen',
      body: 'Drag to rapidly mutate the genes of creatures under the brush — directed evolution by hand. The effect is invisible at first; colour by a trait and wait a few generations to see it.',
      tech: 'Perturbs a few random genes per tick',
    },
    es: {
      title: 'Mutágeno',
      body: 'Arrastra para mutar rápido los genes de las criaturas bajo el pincel — evolución dirigida a mano. El efecto es invisible al principio; colorea por un rasgo y espera unas generaciones para verlo.',
      tech: 'Perturba unos pocos genes al azar por tick',
    },
  },
  tool_smite: {
    en: {
      title: 'Cataclysm',
      body: 'Drag to instantly wipe out every creature under the brush — make room, or watch the ocean repopulate.',
    },
    es: {
      title: 'Cataclismo',
      body: 'Arrastra para aniquilar al instante toda criatura bajo el pincel — haz sitio, o mira cómo el océano se repuebla.',
    },
  },
  tool_size: {
    en: { title: 'Brush size', body: 'How wide every brush reaches.' },
    es: { title: 'Tamaño del pincel', body: 'Hasta dónde llega cada pincel.' },
  },
};

let tipEl: HTMLDivElement | null = null;
let titleEl: HTMLDivElement;
let bodyEl: HTMLDivElement;
let techEl: HTMLDivElement;
let showTimer = 0;

function ensureEl(): void {
  if (tipEl) return;
  tipEl = document.createElement('div');
  tipEl.className = 'pg-panel';
  tipEl.style.cssText =
    'position:fixed;display:none;max-width:248px;padding:9px 11px;z-index:2000;' +
    'pointer-events:none;line-height:1.45;';
  titleEl = document.createElement('div');
  titleEl.style.cssText = 'font:600 12px var(--font-ui);color:var(--glow-cyan);margin-bottom:3px;';
  bodyEl = document.createElement('div');
  bodyEl.style.cssText = 'font:12px var(--font-ui);color:var(--ink);';
  techEl = document.createElement('div');
  techEl.style.cssText = 'font:11px var(--font-mono);color:var(--ink-dim);margin-top:5px;';
  tipEl.append(titleEl, bodyEl, techEl);
  document.body.appendChild(tipEl);
}

function show(el: HTMLElement, key: string): void {
  const entry = TIPS[key];
  if (!entry) return;
  ensureEl();
  const tip = entry[getLang()];
  titleEl.textContent = tip.title;
  bodyEl.textContent = tip.body;
  techEl.textContent = tip.tech ?? '';
  techEl.style.display = tip.tech ? 'block' : 'none';
  tipEl!.style.display = 'block';

  // Position above the element by default, flipping below if it would clip the top.
  const r = el.getBoundingClientRect();
  const tr = tipEl!.getBoundingClientRect();
  let top = r.top - tr.height - 8;
  if (top < 8) top = r.bottom + 8;
  let left = r.left + r.width / 2 - tr.width / 2;
  left = Math.max(8, Math.min(window.innerWidth - tr.width - 8, left));
  tipEl!.style.left = `${left}px`;
  tipEl!.style.top = `${top}px`;
}

function hide(): void {
  window.clearTimeout(showTimer);
  if (tipEl) tipEl.style.display = 'none';
}

/** Show a styled, localised tooltip for `el` on hover. `key` indexes TIPS. */
export function attachTooltip(el: HTMLElement, key: string): void {
  el.removeAttribute('title'); // avoid the native tooltip doubling up
  el.addEventListener('pointerenter', () => {
    window.clearTimeout(showTimer);
    showTimer = window.setTimeout(() => show(el, key), 320);
  });
  el.addEventListener('pointerleave', hide);
  el.addEventListener('pointerdown', hide);
}
