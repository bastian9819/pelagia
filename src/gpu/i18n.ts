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
  highlight: { en: 'highlight lineage', es: 'resaltar linaje' },
  observatory: { en: 'observatory', es: 'observatorio' },
  ph_history: { en: 'history', es: 'historia' },
  ph_title: { en: 'evolutionary history', es: 'historia evolutiva' },
  ph_note: {
    en: 'Each band is a clade; its thickness is how many creatures it has. Watch clades rise, take over and go extinct over time.',
    es: 'Cada banda es un clado; su grosor es cuántas criaturas tiene. Mira a los clados surgir, dominar y extinguirse con el tiempo.',
  },
  ph_time: { en: 'time →', es: 'tiempo →' },
  ph_empty: { en: 'gathering history…', es: 'reuniendo historia…' },
  obs_world: { en: 'the world', es: 'el mundo' },
  obs_lineages: { en: 'lineages over time', es: 'linajes en el tiempo' },
  obs_watched: { en: 'tracked creatures', es: 'criaturas seguidas' },
  obs_alive: { en: 'creatures', es: 'criaturas' },
  obs_food: { en: 'food', es: 'comida' },
  obs_diversity: { en: 'lineages', es: 'linajes' },
  obs_meanEnergy: { en: 'mean energy', es: 'energía media' },
  obs_population: { en: 'population & food', es: 'población y comida' },
  obs_strategy: { en: 'strategy mix (sampled)', es: 'mezcla de estrategias (muestreo)' },
  obs_energyBand: { en: 'energy (min · mean · max)', es: 'energía (mín · media · máx)' },
  obs_watchHint: {
    en: 'Click a creature, then “track” in its brain panel to follow it here.',
    es: 'Haz clic en una criatura y pulsa “seguir” en su panel de cerebro.',
  },
  obs_watchFull: { en: 'tracking limit reached (8)', es: 'límite de seguimiento (8)' },
  track: { en: 'track', es: 'seguir' },
  tracking: { en: 'tracking', es: 'siguiendo' },
  obs_energyLine: { en: 'energy', es: 'energía' },
  obs_speedLine: { en: 'speed', es: 'velocidad' },
  nar_pop: { en: 'Population', es: 'Población' },
  nar_rising: { en: 'rising', es: 'al alza' },
  nar_stable: { en: 'holding steady', es: 'estable' },
  nar_falling: { en: 'falling', es: 'a la baja' },
  nar_predActive: { en: 'Predation active', es: 'Depredación activa' },
  nar_predOff: { en: 'no predation', es: 'sin depredación' },
  nar_strategy: { en: 'Dominant strategy', es: 'Estrategia dominante' },
  nar_complexity: { en: 'mean brain', es: 'cerebro medio' },
  nar_size: { en: 'mean size', es: 'tamaño medio' },
  nar_day: { en: 'day', es: 'día' },
  nar_night: { en: 'night', es: 'noche' },
  nar_dusk: { en: 'dawn/dusk', es: 'amanecer/atardecer' },
  obs_age: { en: 'seen for', es: 'visto durante' },
  obs_ticks: { en: 'ticks', es: 'ticks' },
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
  bv_policy: { en: 'turn vs bearing', es: 'giro según rumbo' },
  bv_food: { en: 'food', es: 'comida' },
  bv_nbr: { en: 'neighbour', es: 'vecino' },
  bv_listens: { en: 'neurons attend to', es: 'las neuronas atienden a' },
  eeg_title: {
    en: 'decision tape (senses → decisions)',
    es: 'tira de decisiones (sentidos → decisiones)',
  },
  turnRight: { en: 'right ▶', es: 'derecha ▶' },
  turnLeft: { en: '◀ left', es: '◀ izquierda' },
  straight: { en: '— straight', es: '— recto' },
  deceased: { en: 'deceased', es: 'muerta' },
  lineageWord: { en: 'lineage', es: 'linaje' },
  energyWord: { en: 'energy', es: 'energía' },
  speedWord: { en: 'speed', es: 'velocidad' },
  sizeWord: { en: 'size', es: 'tamaño' },
  neurons: { en: 'neurons', es: 'neuronas' },
  godMode: { en: 'god mode', es: 'modo dios' },
  reset: { en: 'reset', es: 'reiniciar' },
  g_food: { en: 'food spawn', es: 'aparición de comida' },
  g_mutRate: { en: 'mutation rate', es: 'tasa de mutación' },
  g_mutSize: { en: 'mutation size', es: 'tamaño de mutación' },
  g_speed: { en: 'max speed', es: 'velocidad máx.' },
  g_speciation: { en: 'speciation rate', es: 'tasa de especiación' },
  ph_tree: {
    en: 'family tree (who descends from whom)',
    es: 'árbol genealógico (quién desciende de quién)',
  },
  g_agility: { en: 'agility', es: 'agilidad' },
  g_metabolism: { en: 'metabolism', es: 'metabolismo' },
  g_foodEnergy: { en: 'food energy', es: 'energía de comida' },
  g_reproAt: { en: 'reproduce at', es: 'se reproduce a' },
  g_predation: { en: 'predation', es: 'depredación' },
  g_predMargin: { en: 'predation margin', es: 'margen de depredación' },
  g_offspring: { en: 'offspring energy', es: 'energía de la cría' },
  g_moveCost: { en: 'move cost', es: 'coste de moverse' },
  g_patchiness: { en: 'food patchiness', es: 'agrupación de comida' },
  g_bigFood: { en: 'big-food value', es: 'valor comida grande' },
  g_dayNight: { en: 'day/night swing', es: 'oscilación día/noche' },
  g_dayLength: { en: 'day length', es: 'duración del día' },
  obs_predation: { en: 'predation', es: 'depredación' },
  tr_aggr: { en: 'aggression', es: 'agresión' },
  loading: { en: 'summoning the ocean…', es: 'invocando el océano…' },
  desc_chase: { en: 'chases food head-on', es: 'persigue la comida de frente' },
  desc_steer: { en: 'steers toward food (cautious)', es: 'gira hacia la comida (cauta)' },
  desc_straight: { en: 'fast straight-swimmer', es: 'nada recto y rápido' },
  desc_away: { en: 'turns away from food', es: 'se aleja de la comida' },
  desc_erratic: { en: 'erratic / undirected', es: 'errática / sin rumbo' },
  desc_circler: { en: 'circles in place', es: 'gira en círculos' },
  desc_ambush: { en: 'lurks, then darts at food', es: 'acecha y embiste la comida' },
  desc_predator: { en: 'hunts other creatures', es: 'caza a otras criaturas' },
  desc_skittish: { en: 'flees from others', es: 'huye de las demás' },
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
      '• <b>Predation</b>: creatures hunt each other — a bigger one eats a smaller ' +
      'neighbour on contact, so predator and prey lineages emerge.<br>' +
      '• <b>Evolving brains</b>: each brain can switch hidden neurons on or off across ' +
      'generations, so its complexity itself evolves.<br>' +
      '• <b>God mode</b>: change the world (food, mutation, predation…) and watch ' +
      'evolution respond.<br>' +
      '• <b>Observatory</b> (📊): charts of the world over time, lineage histories, and ' +
      'any creatures you track.<br>' +
      '• <b>🎨 Colour</b> the ocean by a trait (size, neurons, energy…) to watch evolution ' +
      'sweep across it; <b>⏭ step</b> one tick and <b>0.1×</b> slow-mo to study a decision.<br>' +
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
      '• <b>Depredación</b>: las criaturas se cazan entre sí — la más grande se come a ' +
      'una vecina más pequeña al contacto, así surgen linajes depredadores y presa.<br>' +
      '• <b>Cerebros que evolucionan</b>: cada cerebro puede encender o apagar neuronas ' +
      'ocultas entre generaciones, así que su complejidad también evoluciona.<br>' +
      '• <b>Modo dios</b>: cambia el mundo (comida, mutación, depredación…) y mira cómo ' +
      'responde la evolución.<br>' +
      '• <b>Observatorio</b> (📊): gráficas del mundo en el tiempo, historia de los ' +
      'linajes y las criaturas que sigas.<br>' +
      '• <b>🎨 Colorea</b> el océano por un rasgo (tamaño, neuronas, energía…) para ver la ' +
      'evolución recorrerlo; <b>⏭ paso</b> a paso y cámara lenta <b>0.1×</b> para estudiar una ' +
      'decisión.<br>' +
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
