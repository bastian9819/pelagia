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
  // --- Panels ---
  panel_brain: {
    en: {
      title: 'Creature brain',
      body: "This creature's real neural network firing live. Left column = its senses, middle = hidden neurons, right = its decisions (turn, thrust, attack). Nobody scripted it — the weights are in its genome.",
      tech: 'Node colour = activation: cyan positive, magenta negative; brightness = strength.',
    },
    es: {
      title: 'Cerebro de la criatura',
      body: 'La red neuronal real de esta criatura disparándose en vivo. Columna izquierda = sus sentidos, centro = neuronas ocultas, derecha = sus decisiones (giro, empuje, ataque). Nadie la programó — los pesos están en su genoma.',
      tech: 'Color del nodo = activación: cian positiva, magenta negativa; brillo = intensidad.',
    },
  },
  brain_policy: {
    en: {
      title: 'Steering policy',
      body: 'How this brain turns depending on where it senses plankton, big food or a neighbour — read off its genome, not its current state.',
      tech: 'X axis = relative bearing (−180°…180°); Y axis = chosen turn.',
    },
    es: {
      title: 'Política de giro',
      body: 'Cómo gira este cerebro según dónde percibe plancton, comida grande o un vecino — leído de su genoma, no de su estado actual.',
      tech: 'Eje X = rumbo relativo (−180°…180°); eje Y = giro decidido.',
    },
  },
  brain_eeg: {
    en: {
      title: 'Decision tape',
      body: 'What this creature senses and decides, scrolling over time — one sample per tick, so it freezes when paused and advances with the step button.',
      tech: 'Top lanes = senses, bottom lanes = decisions (turn, thrust, attack).',
    },
    es: {
      title: 'Tira de decisiones',
      body: 'Lo que esta criatura siente y decide, desplazándose en el tiempo — una muestra por tick, así que se congela en pausa y avanza con el botón de paso.',
      tech: 'Carriles de arriba = sentidos, de abajo = decisiones (giro, empuje, ataque).',
    },
  },
  panel_lineage: {
    en: {
      title: 'Lineages',
      body: 'Families of creatures sharing a colour. "Dominant" = the most numerous; "distinct" = the most unusual in behaviour. The bars show each family\'s evolved brain traits.',
      tech: 'seek / big-food / aggression: −1 (avoids) … +1 (steers strongly toward); neurons = active of 10. Cyan = positive, magenta = negative.',
    },
    es: {
      title: 'Linajes',
      body: 'Familias de criaturas que comparten color. "Dominantes" = las más numerosas; "distintas" = las de comportamiento más singular. Las barras muestran los rasgos del cerebro evolucionado de cada familia.',
      tech: 'busca / comida grande / agresión: −1 (evita) … +1 (se dirige con fuerza); neuronas = activas de 10. Turquesa = positivo, magenta = negativo.',
    },
  },
  // --- God-mode parameters ---
  g_speed: {
    en: { title: 'Max speed', body: 'How fast creatures can swim at full thrust.' },
    es: { title: 'Velocidad máx.', body: 'Lo rápido que pueden nadar a empuje máximo.' },
  },
  g_agility: {
    en: { title: 'Agility', body: 'How sharply a creature can turn each tick.' },
    es: { title: 'Agilidad', body: 'Cuánto puede girar una criatura en cada tick.' },
  },
  g_eatRange: {
    en: { title: 'Eat range', body: 'How close a creature must be to eat a food pellet.' },
    es: { title: 'Rango de comer', body: 'Cuánto debe acercarse una criatura para comer.' },
  },
  g_current: {
    en: {
      title: 'Current',
      body: 'Strength of the ocean current that drags every creature and food along.',
      tech: 'High values sweep creatures into fast streaks',
    },
    es: {
      title: 'Corriente',
      body: 'Fuerza de la corriente oceánica que arrastra a criaturas y comida.',
      tech: 'Valores altos las arrastran en estelas rápidas',
    },
  },
  g_phero: {
    en: {
      title: 'Pheromones',
      body: 'How much trail each creature leaves behind. Brains can sense and follow trails.',
      tech: 'The "show trails" view reveals the field',
    },
    es: {
      title: 'Feromonas',
      body: 'Cuánto rastro deja cada criatura. El cerebro puede percibir y seguir rastros.',
      tech: 'El modo "mostrar rastros" revela el campo',
    },
  },
  g_food: {
    en: {
      title: 'Food spawn',
      body: 'How much food appears per tick — the main lever on how many creatures the ocean can feed.',
    },
    es: {
      title: 'Aparición de comida',
      body: 'Cuánta comida aparece por tick — la palanca principal de cuántas criaturas puede alimentar el océano.',
    },
  },
  g_foodEnergy: {
    en: { title: 'Food energy', body: 'How much energy a creature gains from eating one pellet.' },
    es: { title: 'Energía de comida', body: 'Cuánta energía gana una criatura al comer un trozo.' },
  },
  g_patchiness: {
    en: {
      title: 'Food patchiness',
      body: 'How clumped food is: spread out (low) or in tight blooms (high).',
    },
    es: {
      title: 'Agrupación de comida',
      body: 'Cómo se reparte la comida: dispersa (bajo) o en floraciones densas (alto).',
    },
  },
  g_bigFood: {
    en: {
      title: 'Big-food value',
      body: 'How much more energy a rare golden big-food pellet gives vs plankton.',
    },
    es: {
      title: 'Valor comida grande',
      body: 'Cuánta más energía da la rara comida grande dorada frente al plancton.',
    },
  },
  g_bigFoodAmt: {
    en: {
      title: 'Big-food amount',
      body: 'What fraction of food is the rare big kind (0 = none).',
    },
    es: {
      title: 'Cantidad comida grande',
      body: 'Qué fracción de la comida es del tipo grande raro (0 = nada).',
    },
  },
  g_carrion: {
    en: {
      title: 'Carrion',
      body: 'Chance a dead creature leaves food where it fell, feeding scavengers.',
    },
    es: {
      title: 'Carroña',
      body: 'Probabilidad de que una criatura muerta deje comida donde cae, alimentando carroñeros.',
    },
  },
  g_mutRate: {
    en: {
      title: 'Mutation rate',
      body: 'How often each gene changes when a creature is born. More = faster but messier evolution.',
    },
    es: {
      title: 'Tasa de mutación',
      body: 'Con qué frecuencia cambia cada gen al nacer. Más = evolución más rápida pero más caótica.',
    },
  },
  g_mutSize: {
    en: { title: 'Mutation size', body: 'How big each mutation is when it happens.' },
    es: { title: 'Tamaño de mutación', body: 'Cómo de grande es cada mutación cuando ocurre.' },
  },
  g_reproAt: {
    en: {
      title: 'Reproduce at',
      body: 'How much energy a creature must store before it can have offspring.',
    },
    es: {
      title: 'Se reproduce a',
      body: 'Cuánta energía debe acumular una criatura para tener cría.',
    },
  },
  g_offspring: {
    en: { title: 'Offspring energy', body: "How much of a parent's energy goes to each newborn." },
    es: { title: 'Energía de la cría', body: 'Cuánta energía del progenitor recibe cada cría.' },
  },
  g_offspringSpread: {
    en: { title: 'Offspring spread', body: 'How far from the parent a newborn appears.' },
    es: { title: 'Dispersión de cría', body: 'A qué distancia del progenitor aparece la cría.' },
  },
  g_speciation: {
    en: {
      title: 'Speciation rate',
      body: 'Chance a newborn founds a brand-new lineage (new colour) — makes the family tree branch.',
    },
    es: {
      title: 'Tasa de especiación',
      body: 'Probabilidad de que una cría funde un linaje nuevo (color nuevo) — ramifica el árbol genealógico.',
    },
  },
  g_sexual: {
    en: {
      title: 'Sexual repro',
      body: "Chance a birth mixes two parents' genes instead of cloning one.",
    },
    es: {
      title: 'Reproducción sexual',
      body: 'Probabilidad de que un nacimiento mezcle genes de dos progenitores en vez de clonar uno.',
    },
  },
  g_mate: {
    en: {
      title: 'Mate choice',
      body: 'How strongly creatures prefer brighter partners when mating — turns glow into a sexually-selected ornament.',
    },
    es: {
      title: 'Elección de pareja',
      body: 'Cuánto prefieren parejas más brillantes al aparearse — convierte el brillo en un ornamento de selección sexual.',
    },
  },
  g_metabolism: {
    en: { title: 'Metabolism', body: 'Energy a creature burns each tick just to stay alive.' },
    es: {
      title: 'Metabolismo',
      body: 'Energía que gasta una criatura cada tick solo por seguir viva.',
    },
  },
  g_moveCost: {
    en: {
      title: 'Move cost',
      body: 'Extra energy spent proportional to how fast a creature swims.',
    },
    es: {
      title: 'Coste de moverse',
      body: 'Energía extra gastada en proporción a lo rápido que nada.',
    },
  },
  g_turnCost: {
    en: { title: 'Turn cost', body: "Energy spent on turning, so agile steering isn't free." },
    es: {
      title: 'Coste de giro',
      body: 'Energía gastada al girar, para que maniobrar no sea gratis.',
    },
  },
  g_glowCost: {
    en: { title: 'Glow cost', body: 'Energy cost of bioluminescence (0 = glowing is free).' },
    es: {
      title: 'Coste de brillo',
      body: 'Coste energético de la bioluminiscencia (0 = brillar es gratis).',
    },
  },
  g_thermal: {
    en: {
      title: 'Thermal contrast',
      body: 'How much living in the wrong temperature band costs energy, pushing lineages to adapt to regions.',
    },
    es: {
      title: 'Contraste térmico',
      body: 'Cuánta energía cuesta vivir en la banda de temperatura equivocada, empujando a los linajes a adaptarse por zonas.',
    },
  },
  g_predation: {
    en: {
      title: 'Predation',
      body: 'How much energy a predator gains from eating prey (0 = no predation at all).',
    },
    es: {
      title: 'Depredación',
      body: 'Cuánta energía gana un depredador al comerse una presa (0 = sin depredación).',
    },
  },
  g_predMargin: {
    en: {
      title: 'Predation margin',
      body: 'How much bigger a hunter must be than its prey to eat it.',
    },
    es: {
      title: 'Margen de depredación',
      body: 'Cuánto más grande debe ser un cazador que su presa para comerla.',
    },
  },
  g_attackCost: {
    en: {
      title: 'Attack cost',
      body: 'Energy spent each tick a creature has its attack on, so lunging has a price.',
    },
    es: {
      title: 'Coste de ataque',
      body: 'Energía gastada cada tick con el ataque activo, para que embestir tenga precio.',
    },
  },
  g_toxin: {
    en: {
      title: 'Toxicity',
      body: 'How much energy a predator loses for eating a toxic prey — makes toxicity a real defence.',
    },
    es: {
      title: 'Toxicidad',
      body: 'Cuánta energía pierde un depredador al comerse una presa tóxica — hace de la toxicidad una defensa real.',
    },
  },
  g_dayNight: {
    en: {
      title: 'Day / night',
      body: 'Strength of the day/night cycle: food swings between boom and bust, and the ocean brightens and dims.',
    },
    es: {
      title: 'Día / noche',
      body: 'Fuerza del ciclo día/noche: la comida oscila entre auge y escasez, y el océano se ilumina y se oscurece.',
    },
  },
  g_dayLength: {
    en: { title: 'Day length', body: 'How long one day/night cycle lasts, in ticks.' },
    es: { title: 'Duración del día', body: 'Cuánto dura un ciclo día/noche, en ticks.' },
  },
  // --- Mechanism toggles (on/off switches) ---
  tog_predation: {
    en: {
      title: 'Predation',
      body: 'Turn hunting on or off. When on, bigger creatures can eat smaller neighbours, so predator and prey lineages emerge.',
    },
    es: {
      title: 'Depredación',
      body: 'Activa o desactiva la caza. Con ella, las más grandes pueden comerse a vecinas más pequeñas, y surgen linajes depredadores y presa.',
    },
  },
  tog_daynight: {
    en: {
      title: 'Day / night',
      body: 'Turn the day/night cycle on or off. Food booms by day and grows scarce by night, so the population rises and falls.',
    },
    es: {
      title: 'Día / noche',
      body: 'Activa o desactiva el ciclo día/noche. La comida abunda de día y escasea de noche, así la población sube y baja.',
    },
  },
  tog_speciation: {
    en: {
      title: 'Speciation',
      body: 'Turn lineage branching on or off. When on, some newborns found a new family (new colour), so the tree of life branches over time.',
    },
    es: {
      title: 'Especiación',
      body: 'Activa o desactiva la ramificación de linajes. Con ella, algunas crías fundan una familia nueva (color nuevo) y el árbol de la vida se ramifica.',
    },
  },
  tog_bigfood: {
    en: {
      title: 'Big food',
      body: 'Turn the rare golden big-food on or off. It appears in tight blooms worth far more energy than plankton.',
    },
    es: {
      title: 'Comida grande',
      body: 'Activa o desactiva la rara comida grande dorada. Aparece en floraciones densas que valen mucha más energía que el plancton.',
    },
  },
  tog_current: {
    en: {
      title: 'Current',
      body: 'Turn the ocean current on or off. It drags creatures and food into drifting gyres.',
    },
    es: {
      title: 'Corriente',
      body: 'Activa o desactiva la corriente oceánica. Arrastra criaturas y comida en remolinos a la deriva.',
    },
  },
  tog_carrion: {
    en: {
      title: 'Carrion',
      body: 'Turn carrion on or off. When on, dead creatures leave food where they fell, feeding scavengers.',
    },
    es: {
      title: 'Carroña',
      body: 'Activa o desactiva la carroña. Con ella, las criaturas muertas dejan comida donde caen, alimentando carroñeros.',
    },
  },
  // --- Scenario presets ---
  pre_eden: {
    en: {
      title: 'Eden',
      body: 'A gentle paradise: abundant food, no predation, cheap living. The ocean fills up calmly.',
    },
    es: {
      title: 'Edén',
      body: 'Un paraíso apacible: comida abundante, sin depredación, vida barata. El océano se llena con calma.',
    },
  },
  pre_famine: {
    en: {
      title: 'Famine',
      body: 'Scarce food, costly living and predation — a harsh world that selects hard.',
    },
    es: {
      title: 'Hambruna',
      body: 'Comida escasa, vida cara y depredación — un mundo duro que selecciona con fuerza.',
    },
  },
  pre_carnage: {
    en: {
      title: 'Carnage',
      body: 'Predation maxed with an easy size advantage — hunters everywhere.',
    },
    es: {
      title: 'Carnicería',
      body: 'Depredación al máximo con ventaja de tamaño fácil — cazadores por todas partes.',
    },
  },
  pre_soup: {
    en: {
      title: 'Primordial soup',
      body: 'High mutation and speciation — fast, chaotic evolution with many new lineages.',
    },
    es: {
      title: 'Sopa primordial',
      body: 'Mutación y especiación altas — evolución rápida y caótica con muchos linajes nuevos.',
    },
  },
  pre_night: {
    en: {
      title: 'Eternal night',
      body: 'A strong, long day/night cycle — dramatic booms and busts.',
    },
    es: {
      title: 'Noche eterna',
      body: 'Un ciclo día/noche fuerte y largo — auges y caídas dramáticos.',
    },
  },
  pre_titans: {
    en: { title: 'Titans', body: 'Strong predation plus rich big-food — favours large hunters.' },
    es: {
      title: 'Titanes',
      body: 'Depredación fuerte y comida grande abundante — favorece a grandes cazadores.',
    },
  },
  pre_dice: {
    en: {
      title: 'Random world',
      body: 'Roll every setting at once (kept away from instant extinction) for a fresh, shareable ocean.',
    },
    es: {
      title: 'Mundo aleatorio',
      body: 'Tira todos los ajustes a la vez (evitando la extinción inmediata) para un océano nuevo y compartible.',
    },
  },
  // --- Menu panel/visualisation toggles ---
  menu_observatory: {
    en: {
      title: 'Observatory',
      body: 'A full-screen dashboard: charts of the world over time, lineage histories and the creatures you track.',
    },
    es: {
      title: 'Observatorio',
      body: 'Un panel a pantalla completa: gráficas del mundo en el tiempo, historia de los linajes y las criaturas que sigas.',
    },
  },
  menu_history: {
    en: {
      title: 'Evolutionary history',
      body: 'A full-screen view of how lineages rose, dominated and went extinct, plus the family tree.',
    },
    es: {
      title: 'Historia evolutiva',
      body: 'Una vista a pantalla completa de cómo los linajes surgieron, dominaron y se extinguieron, con su árbol genealógico.',
    },
  },
  menu_highlight: {
    en: {
      title: 'Highlight lineage',
      body: "Dim every creature except the selected one's family, so you can follow a single clade through the crowd.",
    },
    es: {
      title: 'Resaltar linaje',
      body: 'Atenúa a todas las criaturas salvo la familia de la seleccionada, para seguir un solo clado entre la multitud.',
    },
  },
  menu_fields: {
    en: {
      title: 'Show biomes',
      body: 'Tint the background by water temperature (blue cold, red warm) to reveal the thermal biomes creatures adapt to.',
    },
    es: {
      title: 'Mostrar biomas',
      body: 'Tiñe el fondo por temperatura del agua (azul frío, rojo cálido) para ver los biomas térmicos a los que se adaptan.',
    },
  },
  menu_follow: {
    en: {
      title: 'Follow camera',
      body: 'Lock the camera onto the selected creature to watch its life up close.',
    },
    es: {
      title: 'Seguir cámara',
      body: 'Fija la cámara en la criatura seleccionada para ver su vida de cerca.',
    },
  },
  menu_currents: {
    en: {
      title: 'Show currents',
      body: 'Reveal the invisible ocean current as animated flow streaks.',
    },
    es: {
      title: 'Mostrar corrientes',
      body: 'Revela la corriente oceánica invisible como estrías de flujo animadas.',
    },
  },
  menu_phero: {
    en: {
      title: 'Show trails',
      body: 'Reveal the pheromone field — the chemical trails creatures lay down and can follow.',
    },
    es: {
      title: 'Mostrar rastros',
      body: 'Revela el campo de feromonas — los rastros químicos que las criaturas dejan y pueden seguir.',
    },
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
