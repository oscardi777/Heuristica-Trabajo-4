import os
import math
import time
import random
import pandas as pd

# ─────────────────────────────────────────────
# Archivos
# ─────────────────────────────────────────────
INSTANCES_DIR    = "NWJSSP Instances"
OUTPUT_FILE_EPR  = "resultados\\NWJSSP_OADG_EPR(VND).xlsx"

# ─────────────────────────────────────────────
# Parámetros BRKGA (base genética)
# ─────────────────────────────────────────────
POP_SIZE            = 100
ELITE_SIZE          = 25
MUT_PROB            = 0.20
BIAS_FATHER         = 0.75
SIZE_MUTATE_KEYS    = 0.15
TOURNAMENT_K        = 2
MAX_GEN_NO_IMPROVE  = 250
MAX_TIME            = 3600

# ─────────────────────────────────────────────
# Parámetros EPR
# ─────────────────────────────────────────────
EPR_FREQ            = 10
ELITE_POOL_SIZE     = 5
VND_TIME_LIMIT      = 100

# ─────────────────────────────────────────────
# Instancias a procesar
# ─────────────────────────────────────────────
INSTANCES = [
    #"ft06.txt",           "ft06r.txt",
    #"ft10.txt",           "ft10r.txt",
    #"ft20.txt",           "ft20r.txt",
    #"tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    #"tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
]

random.seed(42)


# ═════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═════════════════════════════════════════════

class Operation:
    """Representa una operación de un job: máquina asignada y tiempo de procesamiento."""
    def __init__(self, machine, processing_time):
        self.machine = machine
        self.p = processing_time


class Job:
    """Representa un job con su lista de operaciones y su tiempo de liberación (release)."""
    def __init__(self, operations, release):
        self.operations = operations
        self.release = release


class Machine:
    """
    Representa una máquina con sus intervalos ocupados.
    Permite consultar cuándo termina el último intervalo que empieza antes de un umbral dado,
    lo cual es necesario para la planificación no-wait con tiempos de liberación.
    """
    def __init__(self, id: int):
        self.id = id
        self.intervals: list[tuple[int, int]] = []

    def add(self, b: int, e: int):
        """Registra un intervalo ocupado [b, e) en esta máquina."""
        self.intervals.append((b, e))

    def max_end_before(self, threshold: int):
        """
        Retorna el mayor tiempo de finalización entre todos los intervalos
        cuyo inicio es estrictamente menor que `threshold`.
        Se usa para verificar que una operación no colisione con trabajos previos.
        """
        max_e = 0
        for b, e in self.intervals:
            if b < threshold:
                max_e = max(max_e, e)
        return max_e


# ═════════════════════════════════════════════
# LECTURA DE INSTANCIAS
# ═════════════════════════════════════════════

def read_instance(file):
    """
    Lee una instancia NWJSSP desde un archivo de texto.
    Formato esperado:
        Primera línea: n m  (número de jobs, número de máquinas)
        Siguientes n líneas: m pares (maquina procesamiento) ... release
    Retorna la lista de Jobs y el número de máquinas m.
    """
    with open(file) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [Operation(data[2*i], data[2*i + 1]) for i in range(m)]
            jobs.append(Job(operations, release=data[-1]))
    return jobs, m


# ═════════════════════════════════════════════
# OFFSETS (desplazamientos acumulados de procesamiento)
# ═════════════════════════════════════════════

def precompute_offsets(jobs):
    """
    Precalcula para cada job la lista de desplazamientos acumulados de procesamiento
    entre la primera operación y cada operación u-ésima.
    Esto permite calcular el inicio de cada operación como: start_job + offset[u].
    Retorna una lista de listas de enteros (una por job).
    """
    result = []
    for job in jobs:
        offsets = [0] * len(job.operations)
        total = 0
        for u, op in enumerate(job.operations[:-1]):
            total += op.p
            offsets[u + 1] = total
        result.append(offsets)
    return result


# ═════════════════════════════════════════════
# PLANIFICACIÓN PRECISA (no-wait con release times)
# ═════════════════════════════════════════════

def find_start_preciso(job, machines, offsets):
    """
    Encuentra el menor tiempo de inicio factible para un job dado el estado actual
    de las máquinas (con sus intervalos ya ocupados).
    Itera ajustando el inicio hasta que ninguna operación colisione con trabajos previos.
    Retorna el tiempo de inicio (entero).
    """
    start = job.release
    while True:
        max_candidate = start
        feasible = True
        for u, op in enumerate(job.operations):
            b_op = start + offsets[u]
            e_op = b_op + op.p
            max_ek = machines[op.machine].max_end_before(e_op)
            if max_ek > b_op:
                feasible = False
                candidate = max_ek - offsets[u]
                max_candidate = max(max_candidate, candidate)
        if feasible:
            return start
        start = max_candidate


def schedule_job_preciso(job, machines, job_id, schedule, offsets):
    """
    Planifica un job en las máquinas actualizando sus intervalos ocupados.
    Si `schedule` no es None, registra cada operación en él.
    Retorna el tiempo de completación (finish de la última operación).
    """
    start = find_start_preciso(job, machines, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin = start + offsets[u]
        finish = begin + op.p
        machines[op.machine].add(begin, finish)
        if schedule is not None:
            schedule.append({
                "job": job_id,
                "machine": machines[op.machine].id,
                "operation": u,
                "start": begin,
                "finish": finish
            })
        completion = finish
    return completion


def evaluate_sequence_preciso(sequence, jobs, m, offsets_list, save_schedule=False):
    """
    Evalúa una secuencia de jobs calculando el flujo total (suma de tiempos de completación).
    Usa la planificación precisa no-wait con tiempos de liberación.
    Si `save_schedule=True`, retorna también la lista detallada de operaciones planificadas.
    Retorna: total_flow (int) o (total_flow, schedule) si save_schedule=True.
    """
    machines = [Machine(i) for i in range(m)]
    total_flow = 0
    schedule = [] if save_schedule else None
    for j in sequence:
        total_flow += schedule_job_preciso(jobs[j], machines, j, schedule, offsets_list[j])
    return (total_flow, schedule) if save_schedule else total_flow


# ═════════════════════════════════════════════
# VND — BÚSQUEDA LOCAL CON MÚLTIPLES VECINDARIOS
# ═════════════════════════════════════════════

def generate_insertion_down_neighbors(sequence):
    """
    Genera vecinos mediante inserción hacia adelante:
    extrae el job en posición i y lo inserta en posición j > i.
    Vecindario N1 del VND.
    """
    neighbors = []
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):
            new_seq = sequence[:]
            job = new_seq.pop(i)
            new_seq.insert(j, job)
            neighbors.append(new_seq)
    return neighbors


def generate_swap_neighbors(sequence):
    """
    Genera vecinos mediante intercambio (swap) de dos posiciones i < j.
    Vecindario N2 del VND.
    """
    neighbors = []
    n = len(sequence)
    for i in range(n):
        for j in range(i + 1, n):
            new_seq = sequence[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            neighbors.append(new_seq)
    return neighbors


def generate_insertion_up_neighbors(sequence):
    """
    Genera vecinos mediante inserción hacia atrás:
    extrae el job en posición i y lo inserta en posición j < i.
    Vecindario N3 del VND.
    """
    neighbors = []
    n = len(sequence)
    for i in range(n):
        for j in range(i):
            new_seq = sequence[:]
            job = new_seq.pop(i)
            new_seq.insert(j, job)
            neighbors.append(new_seq)
    return neighbors


VND_NEIGHBORHOODS = [
    generate_insertion_down_neighbors,
    generate_swap_neighbors,
    generate_insertion_up_neighbors,
]


def first_improvement_ls(sequence, current_z, jobs, m, offsets_list, neighborhood_fn, start_time, time_limit):
    """
    Búsqueda local de primera mejora sobre el vecindario dado por `neighborhood_fn`.
    Recorre los vecinos y acepta el primero que mejore la solución actual.
    Se detiene si se supera `time_limit` segundos desde `start_time`.
    Retorna: (mejor_secuencia, mejor_z, mejoró:bool).
    """
    neighbors = neighborhood_fn(sequence)
    for neighbor in neighbors:
        if time.time() - start_time >= time_limit:
            break
        z = evaluate_sequence_preciso(neighbor, jobs, m, offsets_list)
        if z < current_z:
            return neighbor, z, True
    return sequence, current_z, False


def vnd(initial_sequence, jobs, m, offsets_list, start_time, time_limit):
    """
    Variable Neighborhood Descent (VND) con primera mejora.
    Aplica búsqueda local secuencialmente sobre N1, N2, N3.
    Si hay mejora en Nk, regresa a N1. Si no hay mejora, avanza a N(k+1).
    Se detiene al agotar todos los vecindarios o al superar `time_limit`.
    Retorna: (mejor_secuencia, mejor_z).
    """
    B = list(initial_sequence)
    current_z = evaluate_sequence_preciso(B, jobs, m, offsets_list)
    k = 0
    while k < len(VND_NEIGHBORHOODS) and (time.time() - start_time < time_limit):
        candidate, z_candidate, improved = first_improvement_ls(
            B, current_z, jobs, m, offsets_list,
            VND_NEIGHBORHOODS[k], start_time, time_limit
        )
        if improved:
            B = candidate
            current_z = z_candidate
            k = 0   # Reiniciar al primer vecindario
        else:
            k += 1  # Avanzar al siguiente vecindario
    return B, current_z


# ═════════════════════════════════════════════
# DISTANCIA ENTRE SECUENCIAS (Kendall-tau simplificada)
# ═════════════════════════════════════════════

def position_distance(seq_a, seq_b):
    """
    Calcula la distancia posicional entre dos secuencias del mismo conjunto de jobs.
    Para cada job, computa |pos_en_a - pos_en_b| y suma todas las diferencias.
    Sirve como medida de disimilitud para guiar el Path Relinking.
    Retorna un entero >= 0 (0 significa secuencias idénticas).
    """
    n = len(seq_a)
    pos_a = [0] * n
    pos_b = [0] * n
    for idx, j in enumerate(seq_a):
        pos_a[j] = idx
    for idx, j in enumerate(seq_b):
        pos_b[j] = idx
    return sum(abs(pos_a[j] - pos_b[j]) for j in range(n))


# ═════════════════════════════════════════════
# MOVIMIENTO DETERMINISTA DEL CAMINO (Path Relinking)
# ═════════════════════════════════════════════

def path_step(current, target):
    """
    Aplica UN único movimiento determinista que acerca `current` a `target`.
    El movimiento es: encontrar el job que más reduce la distancia posicional
    entre `current` y `target` al ser reubicado en su posición objetivo.

    Estrategia: para cada job en `current` cuya posición difiere de la de `target`,
    moverlo a su posición en `target` dentro de `current` y elegir el que más
    acerque la secuencia a `target` (mayor reducción de distancia).

    Este movimiento es DISTINTO a los movimientos del VND (que son inserción forward,
    swap e inserción backward sin dirección fija). Aquí el movimiento siempre tiene
    como propósito explícito alinear posiciones con `target`, lo que garantiza:
      1. Reproducibilidad: dado el mismo par (current, target) siempre produce el mismo paso.
      2. Convergencia: la distancia a `target` estrictamente disminuye en cada paso.
      3. No redundancia: el movimiento no coincide conceptualmente con VND porque
         está dirigido (guiado por `target`), no es una exploración ciega del vecindario.

    Retorna: nueva secuencia (lista) con el movimiento aplicado.
    """
    n = len(current)

    # Posiciones de cada job en current y en target
    pos_current = [0] * n
    pos_target  = [0] * n
    for idx, j in enumerate(current):
        pos_current[j] = idx
    for idx, j in enumerate(target):
        pos_target[j] = idx

    # Jobs cuya posición en current difiere de la de target
    differing = [j for j in range(n) if pos_current[j] != pos_target[j]]
    if not differing:
        return current[:]  # Ya son iguales

    # Elegir el job cuyo reposicionamiento reduce más la distancia total
    # (greedy: mayor reducción de distancia posicional)
    best_seq  = None
    best_gain = -1

    for job in differing:
        old_pos = pos_current[job]
        new_pos = pos_target[job]

        # Construir la nueva secuencia: extraer job y reinsertarlo en new_pos
        new_seq = current[:]
        new_seq.pop(old_pos)
        new_seq.insert(new_pos, job)

        # Calcular ganancia: cuánto reduce la distancia a target
        gain = position_distance(current, target) - position_distance(new_seq, target)
        if gain > best_gain:
            best_gain = gain
            best_seq  = new_seq

    return best_seq if best_seq is not None else current[:]


# ═════════════════════════════════════════════
# PATH RELINKING — GENERACIÓN DEL CAMINO
# ═════════════════════════════════════════════

def path_relinking_with_vnd(source, target, jobs, m, offsets_list, start_time, time_limit):
    """
    Genera el camino de soluciones intermedias entre `source` y `target` usando
    movimientos deterministas (path_step), y aplica VND sobre cada solución intermedia.

    Flujo:
      1. Parte de `source` y va acercándose a `target` paso a paso.
      2. En cada punto intermedio del camino aplica VND para mejorar la solución.
      3. Registra la mejor solución encontrada en todo el camino.

    Por qué VND en el camino NO lleva de vuelta a source o target:
      - Los movimientos del path_step cambian la secuencia de forma DIRIGIDA hacia target,
        creando soluciones intermedias que no son óptimos locales de VND.
      - Al aplicar VND sobre estas soluciones intermedias, se exploran regiones del espacio
        de búsqueda que son DISTINTAS de los óptimos locales de source y target,
        pues el punto de partida del VND está en una zona diferente del espacio.
      - La combinación camino-dirigido + VND actúa como intensificación en zonas
        intermedias del espacio de permutaciones que de otro modo no se explorarían.

    Retorna: (mejor_secuencia, mejor_z) encontrada en todo el camino.
    """
    current_seq = source[:]
    current_z   = evaluate_sequence_preciso(current_seq, jobs, m, offsets_list)
    best_seq    = current_seq[:]
    best_z      = current_z

    steps = 0
    while position_distance(current_seq, target) > 0:
        if time.time() - start_time >= time_limit:
            break

        # Paso determinista: acercar current hacia target
        current_seq = path_step(current_seq, target)
        steps += 1

        # Aplicar VND sobre la solución intermedia del camino
        # Tiempo limitado para no agotar el presupuesto total
        vnd_seq, vnd_z = vnd(
            current_seq, jobs, m, offsets_list,
            start_time, min(time_limit, time.time() - start_time + VND_TIME_LIMIT)
        )

        # Registrar la mejor solución del camino
        if vnd_z < best_z:
            best_z   = vnd_z
            best_seq = vnd_seq[:]

        # Actualizar current con la solución mejorada por VND solo si mejoró,
        # para continuar el camino desde un punto potencialmente mejor
        if vnd_z < current_z:
            current_seq = vnd_seq[:]
            current_z   = vnd_z

    return best_seq, best_z


# ═════════════════════════════════════════════
# POOL DE ÉLITE (para EPR)
# ═════════════════════════════════════════════

class ElitePool:
    """
    Mantiene un conjunto de las mejores soluciones encontradas durante la búsqueda.
    Garantiza que todas las soluciones del pool sean distintas entre sí (por posición)
    y reemplaza la peor si llega una mejor.
    Se usa para seleccionar los pares (source, target) del Path Relinking.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.solutions: list[tuple[int, list]] = []  # (objective, sequence)

    def add(self, sequence: list, objective: int):
        """
        Intenta agregar una solución al pool.
        Se agrega si:
          a) El pool no está lleno, o
          b) Mejora a la peor solución del pool.
        Además, evita duplicados exactos.
        """
        # Verificar si ya existe en el pool
        for obj, seq in self.solutions:
            if seq == sequence:
                return  # Duplicado exacto, ignorar

        if len(self.solutions) < self.max_size:
            self.solutions.append((objective, sequence[:]))
            self.solutions.sort(key=lambda x: x[0])
        else:
            # Reemplazar la peor si la nueva es mejor
            worst_obj, _ = self.solutions[-1]
            if objective < worst_obj:
                self.solutions[-1] = (objective, sequence[:])
                self.solutions.sort(key=lambda x: x[0])

    def best(self):
        """Retorna la mejor solución del pool como (objective, sequence)."""
        return self.solutions[0] if self.solutions else (float('inf'), [])

    def get_pair(self):
        """
        Selecciona aleatoriamente un par de soluciones distintas del pool
        para usarlas como (source, target) en el Path Relinking.
        Retorna (source_seq, target_seq) o None si el pool tiene menos de 2 soluciones.
        """
        if len(self.solutions) < 2:
            return None
        idx1, idx2 = random.sample(range(len(self.solutions)), 2)
        return self.solutions[idx1][1], self.solutions[idx2][1]

    def __len__(self):
        return len(self.solutions)


# ═════════════════════════════════════════════
# REPRESENTACIÓN DE INDIVIDUOS (BRKGA — Random Keys)
# ═════════════════════════════════════════════

class Individual:
    """
    Individuo del algoritmo genético con codificación de Random Keys.
    La secuencia de jobs se obtiene ordenando los índices por su clave aleatoria asociada.
    Esto garantiza que cualquier vector de reales en [0,1]^n representa una permutación válida.
    """

    def __init__(self, keys: list[float]):
        self.keys = keys
        self.sequence = sorted(range(len(keys)), key=lambda i: keys[i])
        self.objective = None

    def evaluate(self, jobs, m, offsets_list) -> int:
        """
        Evaluación lazy: calcula el objetivo solo si aún no se ha evaluado.
        Retorna el valor de la función objetivo (flujo total).
        """
        if self.objective is None:
            self.objective = evaluate_sequence_preciso(self.sequence, jobs, m, offsets_list)
        return self.objective

    def copy(self):
        """Retorna una copia independiente del individuo (deep copy de keys y objetivo)."""
        ind = Individual(self.keys[:])
        ind.sequence  = self.sequence[:]
        ind.objective = self.objective
        return ind

    def __lt__(self, other):
        if self.objective is None:
            return False
        if other.objective is None:
            return True
        return self.objective < other.objective


# ═════════════════════════════════════════════
# OPERADORES GENÉTICOS
# ═════════════════════════════════════════════

def initialize_population(pop_size: int, n: int) -> list[Individual]:
    """
    Genera una población inicial de `pop_size` individuos con Random Keys
    distribuidas uniformemente en [0, 1).
    """
    return [Individual([random.random() for _ in range(n)]) for _ in range(pop_size)]


def tournament_selection(population: list[Individual], k: int = TOURNAMENT_K) -> Individual:
    """
    Selección por torneo: elige `k` individuos al azar y retorna el de menor objetivo.
    Requiere que todos los individuos ya hayan sido evaluados.
    """
    candidates = random.sample(population, k)
    return min(candidates, key=lambda ind: ind.objective)


def bias_crossover(parent1: Individual, parent2: Individual, bias_father: float = BIAS_FATHER) -> Individual:
    """
    Cruce sesgado (Biased Random Key Crossover):
    Para cada gen i, hereda del mejor padre con probabilidad `bias_father`
    y del otro padre con probabilidad `1 - bias_father`.
    El mejor padre es el de menor objetivo.
    Retorna un nuevo individuo hijo.
    """
    # Asegurarse que parent1 sea el mejor
    if parent2.objective < parent1.objective:
        parent1, parent2 = parent2, parent1

    n = len(parent1.keys)
    child_keys = [
        parent1.keys[i] if random.random() < bias_father else parent2.keys[i]
        for i in range(n)
    ]
    return Individual(child_keys)


def mutate(individual: Individual, mut_prob: float = MUT_PROB) -> Individual:
    """
    Mutación gaussiana sobre Random Keys:
    Con probabilidad `mut_prob` perturba un subconjunto aleatorio de genes
    con ruido gaussiano (media 0, desviación 0.15), manteniendo los valores en [0, 1].
    Garantiza mutar al menos 1 gen cuando se activa.
    Retorna el individuo mutado (o una copia sin cambios si no se activa).
    """
    if random.random() >= mut_prob:
        return individual.copy()

    child = individual.copy()
    n = len(child.keys)
    num_mutations = max(1, int(SIZE_MUTATE_KEYS * n))

    for _ in range(num_mutations):
        i = random.randint(0, n - 1)
        child.keys[i] += random.gauss(0, 0.15)
        child.keys[i] = max(0.0, min(1.0, child.keys[i]))

    child.sequence  = sorted(range(n), key=lambda i: child.keys[i])
    child.objective = None
    return child


# ═════════════════════════════════════════════
# SALIDA A EXCEL
# ═════════════════════════════════════════════

def write_results_to_excel(results, output_file):
    """
    Escribe los resultados de una o más instancias en un archivo Excel.
    Cada instancia ocupa una hoja con:
      - Fila 1: [flujo_total, tiempo_ms]
      - Fila 2: [start_time_job_0, start_time_job_1, ..., start_time_job_n]
    Si el archivo ya existe, reemplaza la hoja correspondiente; si no, lo crea.
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    writer_kwargs = (
        dict(engine="openpyxl", mode="a", if_sheet_exists="replace")
        if os.path.exists(output_file)
        else dict(engine="openpyxl", mode="w")
    )
    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        for sheet_name, (total_flow, compute_time_ms, job_start_times) in results.items():
            df = pd.DataFrame([[total_flow, compute_time_ms], job_start_times])
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
    print(f"  → Resultados guardados en: {output_file}")


# ═════════════════════════════════════════════
# ALGORITMO PRINCIPAL: EPR (Evolutionary Path Relinking)
# ═════════════════════════════════════════════

def epr(jobs, m, offsets_list, start_time,
        pop_size=POP_SIZE, elite_size=ELITE_SIZE,
        mut_prob=MUT_PROB, tournament_k=TOURNAMENT_K,
        max_time=MAX_TIME, max_gen_no_improve=MAX_GEN_NO_IMPROVE,
        bias_father=BIAS_FATHER, epr_freq=EPR_FREQ,
        elite_pool_size=ELITE_POOL_SIZE):
    """
    Evolutionary Path Relinking (EPR) para el problema NWJSSP.

    Descripción general:
    ─────────────────────────────────────────────────────────────────
    EPR combina un algoritmo genético con codificación de Random Keys (BRKGA)
    con Path Relinking como mecanismo de intensificación periódica.

    El algoritmo opera en tres niveles:

      1. BRKGA (evolución genética):
         Mantiene una población de individuos codificados como vectores de reales.
         En cada generación aplica selección por torneo, cruce sesgado y mutación
         gaussiana para generar offspring. Los élites se preservan directamente.

      2. Pool de élite (memoria de buenas soluciones):
         Las mejores soluciones encontradas se acumulan en un pool de tamaño fijo.
         Este pool alimenta el procedimiento de Path Relinking.

      3. Path Relinking con VND (intensificación):
         Cada `epr_freq` generaciones se selecciona un par (source, target) del pool
         y se explora el camino de soluciones intermedias entre ellas.
         En cada punto intermedio se aplica VND para mejorar localmente la solución.
         Los movimientos del camino son deterministas y dirigidos (distintos al VND),
         lo que garantiza explorar regiones nuevas del espacio de permutaciones.

    Criterios de parada:
      - Tiempo total superado (`max_time` segundos).
      - `max_gen_no_improve` generaciones consecutivas sin mejorar el mejor global.

    Parámetros:
      jobs, m, offsets_list : datos de la instancia.
      start_time            : tiempo de inicio (para controlar el reloj).
      pop_size              : tamaño de la población.
      elite_size            : número de individuos élite preservados por generación.
      mut_prob              : probabilidad de mutación.
      tournament_k          : tamaño del torneo para selección.
      max_time              : tiempo máximo en segundos.
      max_gen_no_improve    : generaciones sin mejora para parar.
      bias_father           : sesgo hacia el mejor padre en el cruce.
      epr_freq              : cada cuántas generaciones se ejecuta Path Relinking.
      elite_pool_size       : tamaño máximo del pool de élite para PR.

    Retorna: (mejor_secuencia, mejor_objetivo, tiempo_total_ms).
    """

    n = len(jobs)
    generation = 0
    generations_no_improve = 0
    best_objective = float('inf')
    best_sequence  = None

    # ── Inicialización ──────────────────────────────────────────────
    population = initialize_population(pop_size, n)
    for ind in population:
        ind.evaluate(jobs, m, offsets_list)
    population.sort(key=lambda x: x.objective)

    # Pool de élite para Path Relinking
    pool = ElitePool(elite_pool_size)

    # Cargar el pool con los mejores individuos iniciales
    for ind in population[:elite_size]:
        pool.add(ind.sequence, ind.objective)

    best_sequence  = population[0].sequence[:]
    best_objective = population[0].objective

    print(f"  Gen {generation:>4} | Mejor: {best_objective:>12,} | Pool: {len(pool):>2} | Tiempo: {round((time.time()-start_time)*1000):>7}ms")

    # ── Ciclo evolutivo ─────────────────────────────────────────────
    while time.time() - start_time < max_time:
        generation += 1

        # 1. Preservar élite
        elite = [ind.copy() for ind in population[:elite_size]]

        # 2. Generar offspring mediante cruce y mutación
        offspring = []
        for _ in range(pop_size - elite_size):
            p1 = tournament_selection(population, tournament_k)
            p2 = tournament_selection(population, tournament_k)
            child = bias_crossover(p1, p2, bias_father)
            child = mutate(child, mut_prob)
            offspring.append(child)

        # 3. Evaluar offspring
        for ind in offspring:
            ind.evaluate(jobs, m, offsets_list)

        # 4. Nueva población = élite + offspring, ordenada
        population = elite + offspring
        population.sort(key=lambda x: x.objective)

        # 5. Actualizar pool de élite con los mejores de la generación
        for ind in population[:elite_size]:
            pool.add(ind.sequence, ind.objective)

        # 6. Actualizar mejor global
        improved = False
        if population[0].objective < best_objective:
            best_objective = population[0].objective
            best_sequence  = population[0].sequence[:]
            generations_no_improve = 0
            improved = True
            print(f"  Gen {generation:>4} | Mejor: {best_objective:>12,} ★ | Pool: {len(pool):>2} | Tiempo: {round((time.time()-start_time)*1000):>7}ms")
        else:
            generations_no_improve += 1
            if generation % 25 == 0:
                print(f"  Gen {generation:>4} | Mejor: {best_objective:>12,}   | Pool: {len(pool):>2} | Tiempo: {round((time.time()-start_time)*1000):>7}ms")

        # 7. Path Relinking periódico (cada epr_freq generaciones)
        if generation % epr_freq == 0 and len(pool) >= 2:
            pair = pool.get_pair()
            if pair is not None:
                source_seq, target_seq = pair
                remaining = max_time - (time.time() - start_time)
                if remaining > 0:
                    print(f"  [EPR] Gen {generation:>4} → Path Relinking entre par del pool (dist={position_distance(source_seq, target_seq)})")
                    pr_seq, pr_z = path_relinking_with_vnd(
                        source_seq, target_seq,
                        jobs, m, offsets_list,
                        start_time, max_time
                    )

                    # Agregar resultado de PR al pool
                    pool.add(pr_seq, pr_z)

                    # Verificar si PR encontró el nuevo mejor global
                    if pr_z < best_objective:
                        best_objective = pr_z
                        best_sequence  = pr_seq[:]
                        generations_no_improve = 0
                        print(f"  [EPR] ★ Nueva mejor solución por PR: {best_objective:,}")

                    # Inyectar la solución PR en la población reemplazando el peor individuo
                    # (codificamos la secuencia como keys equiespaciadas según la posición)
                    pr_keys = [0.0] * n
                    for rank, job_id in enumerate(pr_seq):
                        pr_keys[job_id] = rank / n
                    pr_ind = Individual(pr_keys)
                    pr_ind.sequence  = pr_seq[:]
                    pr_ind.objective = pr_z
                    population[-1]   = pr_ind
                    population.sort(key=lambda x: x.objective)

        # 8. Criterio de parada por estancamiento
        if generations_no_improve >= max_gen_no_improve:
            print(f"  → Parada por estancamiento ({max_gen_no_improve} gen sin mejora) en generación {generation}")
            break

    total_time_ms = round((time.time() - start_time) * 1000)
    print(f"\n  Resultado final → Flujo total: {best_objective:,} | Tiempo: {total_time_ms}ms")
    return best_sequence, best_objective, total_time_ms


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    """
    Punto de entrada del script.
    Itera sobre todas las instancias definidas en INSTANCES, ejecuta EPR sobre cada una
    y guarda los resultados en el archivo Excel de salida.
    """
    for inst in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, inst)
        if not os.path.exists(filepath):
            print(f"[SKIP] {inst} — archivo no encontrado")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)
        sheet_name = inst.replace(".txt", "")

        print(f"\n{'═'*60}")
        print(f"[EPR] Procesando instancia: {inst}  (n={n}, m={m})")
        print(f"{'═'*60}")

        offsets_list = precompute_offsets(jobs)

        # Timer inicia ANTES del algoritmo
        t0 = time.time()

        # Ejecutar EPR
        seq_epr, z_epr, elapsed_ms = epr(
            jobs, m, offsets_list, t0,
            pop_size=POP_SIZE,
            elite_size=ELITE_SIZE,
            mut_prob=MUT_PROB,
            tournament_k=TOURNAMENT_K,
            max_time=MAX_TIME,
            max_gen_no_improve=MAX_GEN_NO_IMPROVE,
            bias_father=BIAS_FATHER,
            epr_freq=EPR_FREQ,
            elite_pool_size=ELITE_POOL_SIZE,
        )

        # Obtener tiempos de inicio de cada job (operación 0) para el Excel
        _, schedule = evaluate_sequence_preciso(
            seq_epr, jobs, m, offsets_list, save_schedule=True
        )
        job_start_times = [None] * n
        for op in schedule:
            if op["operation"] == 0:
                job_start_times[op["job"]] = op["start"]

        single_results = {sheet_name: (z_epr, elapsed_ms, job_start_times)}
        write_results_to_excel(single_results, OUTPUT_FILE_EPR)


if __name__ == "__main__":
    main()