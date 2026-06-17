/**
 * Tiny i18n for the UI. All visible text goes through `t(key)`. Components that
 * render dynamically (stats, brain, lineages) pick up the new language on their
 * next update; static labels subscribe via `onLang` to relabel on toggle.
 */
export type Lang = 'en' | 'es';

type Dict = Record<string, { en: string; es: string }>;

const S: Dict = {
  creaturesAlive: { en: 'creatures alive', es: 'criaturas vivas' },
  hint: {
    en: 'drag to pan · scroll to zoom\nclick a creature → its brain',
    es: 'arrastra para mover · rueda para zoom\nclic en una criatura → su cerebro',
  },
  lineages: { en: 'lineages', es: 'linajes' },
  god: { en: 'god', es: 'dios' },
  help: { en: 'help', es: 'ayuda' },
  share: { en: 'share', es: 'compartir' },
  copied: { en: 'link copied', es: 'enlace copiado' },
  restart: { en: 'restart', es: 'reiniciar' },
  dominantLineages: { en: 'dominant lineages', es: 'linajes dominantes' },
  dominant: { en: 'dominant', es: 'dominantes' },
  distinct: { en: 'distinct strategies', es: 'estrategias distintas' },
  tr_seek: { en: 'seek', es: 'busca' },
  tr_forage: { en: 'forage', es: 'forrajeo' },
  tr_cruise: { en: 'cruise', es: 'crucero' },
  sampling: { en: 'sampling…', es: 'muestreando…' },
  creatureBrain: { en: 'creature brain', es: 'cerebro de la criatura' },
  in_foodAhead: { en: 'food ahead', es: 'comida frente' },
  in_foodSide: { en: 'food side', es: 'comida lado' },
  in_foodNear: { en: 'food near', es: 'comida cerca' },
  in_nbrAhead: { en: 'nbr ahead', es: 'vecino frente' },
  in_nbrSide: { en: 'nbr side', es: 'vecino lado' },
  in_nbrNear: { en: 'nbr near', es: 'vecino cerca' },
  in_energy: { en: 'energy', es: 'energía' },
  in_speed: { en: 'speed', es: 'velocidad' },
  out_turn: { en: 'turn', es: 'giro' },
  out_thrust: { en: 'thrust', es: 'empuje' },
  decision: { en: 'decision', es: 'decisión' },
  turnRight: { en: 'right ▶', es: 'derecha ▶' },
  turnLeft: { en: '◀ left', es: '◀ izquierda' },
  straight: { en: '— straight', es: '— recto' },
  deceased: { en: 'deceased', es: 'muerta' },
  lineageWord: { en: 'lineage', es: 'linaje' },
  energyWord: { en: 'energy', es: 'energía' },
  speedWord: { en: 'speed', es: 'velocidad' },
  godMode: { en: 'god mode', es: 'modo dios' },
  reset: { en: 'reset', es: 'reiniciar' },
  g_food: { en: 'food spawn', es: 'aparición comida' },
  g_mutRate: { en: 'mutation rate', es: 'tasa de mutación' },
  g_mutSize: { en: 'mutation size', es: 'tamaño de mutación' },
  g_speed: { en: 'max speed', es: 'velocidad máx.' },
  g_agility: { en: 'agility', es: 'agilidad' },
  g_metabolism: { en: 'metabolism', es: 'metabolismo' },
  g_foodEnergy: { en: 'food energy', es: 'energía de comida' },
  g_reproAt: { en: 'reproduce at', es: 'reproduce a' },
  loading: { en: 'summoning the ocean…', es: 'invocando el océano…' },
  desc_chase: { en: 'chases food head-on', es: 'persigue la comida de frente' },
  desc_steer: { en: 'steers toward food (cautious)', es: 'gira hacia la comida (cauto)' },
  desc_straight: { en: 'fast straight-swimmer', es: 'nada recto y rápido' },
  desc_away: { en: 'turns away from food', es: 'se aleja de la comida' },
  desc_erratic: { en: 'erratic / undirected', es: 'errático / sin rumbo' },
  desc_circler: { en: 'circles in place', es: 'gira en círculos' },
  desc_ambush: { en: 'lurks, then darts at food', es: 'acecha y embiste la comida' },
  fast: { en: 'fast', es: 'rápido' },
  slow: { en: 'slow', es: 'lento' },
  helpTitle: { en: 'what am I looking at?', es: '¿qué estoy viendo?' },
  helpBody: {
    en:
      '<b>PELAGIA</b> is an ocean of artificial life. Each speck is a creature with a ' +
      '<b>real neural network</b> — nobody scripted its behaviour. It senses its world, ' +
      'decides how to move, eats, spends energy, dies, and reproduces with mutation. Over ' +
      'minutes, <b>natural selection</b> makes food-seeking and other behaviours emerge.<br><br>' +
      '• <b>Click a creature</b> to see its brain: left = senses, middle = hidden neurons, ' +
      'right = decisions (turn, thrust). Cyan = positive, magenta = negative.<br>' +
      '• <b>Lineages</b>: descendants of a founder share a colour. The panel ranks the ' +
      'dominant ones and describes their evolved strategy.<br>' +
      '• <b>God mode</b>: change the world (food, mutation…) and watch evolution respond.<br>' +
      '• <b>Drag</b> to pan, <b>scroll</b> to zoom, <b>space</b> to pause.',
    es:
      '<b>PELAGIA</b> es un océano de vida artificial. Cada mota es una criatura con una ' +
      '<b>red neuronal real</b> — nadie programó su comportamiento. Percibe su entorno, ' +
      'decide cómo moverse, come, gasta energía, muere y se reproduce con mutación. En ' +
      'minutos, la <b>selección natural</b> hace emerger la búsqueda de comida y otros ' +
      'comportamientos.<br><br>' +
      '• <b>Haz clic en una criatura</b> para ver su cerebro: izquierda = sentidos, centro = ' +
      'neuronas ocultas, derecha = decisiones (giro, empuje). Cian = positivo, magenta = ' +
      'negativo.<br>' +
      '• <b>Linajes</b>: los descendientes de un fundador comparten color. El panel ordena ' +
      'los dominantes y describe su estrategia evolucionada.<br>' +
      '• <b>Modo dios</b>: cambia el mundo (comida, mutación…) y mira cómo responde la ' +
      'evolución.<br>' +
      '• <b>Arrastra</b> para mover, <b>rueda</b> para zoom, <b>espacio</b> para pausar.',
  },
};

let lang: Lang =
  (localStorage.getItem('pelagia-lang') as Lang | null) ??
  (navigator.language.startsWith('es') ? 'es' : 'en');

const subs = new Set<() => void>();

export function t(key: string): string {
  return S[key]?.[lang] ?? key;
}
export function getLang(): Lang {
  return lang;
}
export function toggleLang(): void {
  lang = lang === 'en' ? 'es' : 'en';
  localStorage.setItem('pelagia-lang', lang);
  for (const fn of subs) fn();
}
export function onLang(fn: () => void): void {
  subs.add(fn);
}
