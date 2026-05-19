import os
import math
import time
import random
import pandas as pd

# ─────────────────────────────────────────────
# Archivos
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE_BKGA = "resultados\\NWJSSP_OADG_NEH(BKGA).xlsx"

# ─────────────────────────────────────────────
# Parámetros del Algoritmo Genético
# ─────────────────────────────────────────────
POP_SIZE       = 200    # Tamaño de la población
ELITE_SIZE     = 20     # Número de individuos élite que sobreviven cada generación
MUT_PROB       = 0.15   # Probabilidad de mutación por individuo
TOURNAMENT_K   = 2      # Tamaño del torneo para selección de padres
MAX_TIME       = 3600   # Tiempo máximo por instancia (segundos)

# ─────────────────────────────────────────────
# Instancias a procesar
# ─────────────────────────────────────────────
INSTANCES = [
    "ft06.txt",           "ft06r.txt",
    "ft10.txt",           "ft10r.txt",
    "ft20.txt",           "ft20r.txt",
    #"tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    #"tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    #"tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    #"tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
]

random.seed(42)


# ─────────────────────────────────────────────
# ESTRUCTURAS DE DATOS (reutilizadas de vnd.py)
# ─────────────────────────────────────────────
class Operation:
    def __init__(self, machine, processing_time):
        self.machine = machine
        self.p = processing_time


class Job:
    def __init__(self, operations, release):
        self.operations = operations
        self.release = release

class Machine:
    def __init__(self, id: int):
        self.id = id
        self.intervals: list[tuple[int, int]] = []

    def add(self, b: int, e: int):
        self.intervals.append((b, e))

    def max_end_before(self, threshold: int):
        max_e = 0
        for b, e in self.intervals:
            if b < threshold:
                max_e = max(max_e, e)
        return max_e


# ─────────────────────────────────────────────
# FUNCIONES DE LECTURA Y EVALUACIÓN (reutilizadas de vnd.py)
# ─────────────────────────────────────────────
def read_instance(file):
    with open(file) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [Operation(data[2*i], data[2*i + 1]) for i in range(m)]
            jobs.append(Job(operations, release=data[-1]))
    return jobs, m


def precompute_offsets(jobs):
    return [
        [0] * len(job.operations) if len(job.operations) <= 1 else
        [0] + [sum(op.p for op in job.operations[:u+1]) for u in range(len(job.operations)-1)]
        for job in jobs
    ]


def compute_offsets(job):
    offsets = [0] * len(job.operations)
    total = 0
    for u, op in enumerate(job.operations[:-1]):
        total += op.p
        offsets[u + 1] = total
    return offsets


def find_start_preciso(job, machines, offsets):
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
    machines = [Machine(i) for i in range(m)]
    total_flow = 0
    schedule = [] if save_schedule else None
    for j in sequence:
        total_flow += schedule_job_preciso(jobs[j], machines, j, schedule, offsets_list[j])
    return (total_flow, schedule) if save_schedule else total_flow


def write_results_to_excel(results, output_file):
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


# ─────────────────────────────────────────────
# REPRESENTACIÓN DE INDIVIDUOS (BKGA)
# ─────────────────────────────────────────────
class Individual:
    """
    Representa un individuo en el algoritmo genético con random keys.

    Atributos:
        keys     : lista de floats en [0, 1) de longitud n
        sequence : permutación de jobs obtenida al ordenar por keys (ascendente)
        objective: valor de la función objetivo (None hasta ser evaluado)
    """

    def __init__(self, keys: list[float]):
        self.keys = keys
        # La secuencia se obtiene ordenando los índices de jobs por su key asociada
        self.sequence = sorted(range(len(keys)), key=lambda i: keys[i])
        self.objective = None

    def evaluate(self, jobs, m, offsets_list) -> int:
        """
        Evaluación lazy: calcula el objetivo solo si aún no ha sido evaluado.
        Retorna el valor de la función objetivo (total flow time).
        """
        if self.objective is None:
            self.objective = evaluate_sequence_preciso(
                self.sequence, jobs, m, offsets_list
            )
        return self.objective

    def copy(self):
        """Retorna una copia independiente del individuo."""
        ind = Individual(self.keys[:])
        ind.objective = self.objective
        return ind

    def __lt__(self, other):
        """Permite comparar individuos por objetivo (menor es mejor)."""
        if self.objective is None:
            return False
        if other.objective is None:
            return True
        return self.objective < other.objective


# ─────────────────────────────────────────────
# INICIALIZACIÓN
# ─────────────────────────────────────────────
def initialize_population(pop_size: int, n: int) -> list[Individual]:
    """
    Genera una población inicial de `pop_size` individuos con random keys
    uniformemente distribuidas en [0, 1).
    """
    return [Individual([random.random() for _ in range(n)]) for _ in range(pop_size)]


# ─────────────────────────────────────────────
# SELECCIÓN: TORNEO BINARIO
# ─────────────────────────────────────────────
def tournament_selection(population: list[Individual], k: int = TOURNAMENT_K) -> Individual:
    """
    Selección por torneo: elige k individuos al azar y retorna el mejor.
    Se asume que todos los individuos ya fueron evaluados.
    """
    candidates = random.sample(population, k)
    return min(candidates, key=lambda ind: ind.objective)


# ─────────────────────────────────────────────
# CRUCE: UNIFORME SOBRE RANDOM KEYS
# ─────────────────────────────────────────────
def uniform_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Cruce uniforme: para cada posición, toma el valor del padre 1 o padre 2
    con probabilidad 0.5 cada uno.
    Retorna un nuevo individuo hijo (offspring).
    """
    n = len(parent1.keys)
    child_keys = [
        parent1.keys[i] if random.random() < 0.5 else parent2.keys[i]
        for i in range(n)
    ]
    return Individual(child_keys)


# ─────────────────────────────────────────────
# MUTACIÓN: SWAP DE DOS POSICIONES EN KEYS
# ─────────────────────────────────────────────
def mutate(individual: Individual, mut_prob: float = MUT_PROB) -> Individual:
    """
    Mutación por swap: con probabilidad `mut_prob`, selecciona dos posiciones
    distintas en la lista de keys y las intercambia.
    Opera sobre una copia del individuo (no modifica el original).
    """
    if random.random() < mut_prob:
        new_keys = individual.keys[:]
        i, j = random.sample(range(len(new_keys)), 2)
        new_keys[i], new_keys[j] = new_keys[j], new_keys[i]
        return Individual(new_keys)   # nuevo individuo con objective=None
    return individual


# ─────────────────────────────────────────────
# ALGORITMO GENÉTICO CON RANDOM KEYS (BKGA)
# ─────────────────────────────────────────────
def bkga(jobs, m, offsets_list, start_time,
         pop_size=POP_SIZE, elite_size=ELITE_SIZE,
         mut_prob=MUT_PROB, tournament_k=TOURNAMENT_K,
         max_time=MAX_TIME):
    """
    Biased Key Genetic Algorithm (BKGA) para el NWJSSP.

    Parámetros:
        jobs         : lista de objetos Job
        m            : número de máquinas
        offsets_list : offsets precomputados por job
        start_time   : tiempo de inicio (time.time())
        pop_size     : tamaño de la población
        elite_size   : número de élites que se preservan por generación
        mut_prob     : probabilidad de mutación
        tournament_k : tamaño del torneo
        max_time     : tiempo máximo en segundos

    Retorna:
        best_sequence : mejor secuencia encontrada
        best_z        : mejor valor de función objetivo
        finish_time   : tiempo de finalización
    """
    n = len(jobs)
    generation = 0

    # ── 1. Inicializar población ──────────────────────────────────────────
    population = initialize_population(pop_size, n)

    # Evaluar toda la población inicial
    for ind in population:
        ind.evaluate(jobs, m, offsets_list)

    # Ordenar de mejor (menor) a peor
    population.sort()

    best = population[0].copy()
    print(f"  Gen {generation:>4} | Mejor objetivo: {best.objective:>12,} | "
          f"Tiempo: {time.time() - start_time:>7.1f}s")

    # ── 2. Ciclo evolutivo ───────────────────────────────────────────────
    while time.time() - start_time < max_time:
        generation += 1

        # ── 2a. Élite: los mejores individuos pasan directamente ──────────
        elite = [ind.copy() for ind in population[:elite_size]]

        # ── 2b. Generar offspring hasta completar la población ────────────
        offspring_size = pop_size - elite_size
        offspring = []

        for _ in range(offspring_size):
            # Selección de padres por torneo
            p1 = tournament_selection(population, tournament_k)
            p2 = tournament_selection(population, tournament_k)

            # Cruce uniforme sobre random keys
            child = uniform_crossover(p1, p2)

            # Mutación
            child = mutate(child, mut_prob)

            offspring.append(child)

        # ── 2c. Evaluar offspring (lazy: solo los no evaluados) ───────────
        for ind in offspring:
            ind.evaluate(jobs, m, offsets_list)

        # ── 2d. Nueva población = élite + mejores offspring ───────────────
        offspring.sort()
        population = elite + offspring
        population.sort()

        # ── 2e. Actualizar mejor solución global ──────────────────────────
        if population[0].objective < best.objective:
            best = population[0].copy()
            elapsed = time.time() - start_time
            print(f"  Gen {generation:>4} | Mejor objetivo: {best.objective:>12,} ★ | "
                  f"Tiempo: {elapsed:>7.1f}s")
        elif generation % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Gen {generation:>4} | Mejor objetivo: {best.objective:>12,}   | "
                  f"Tiempo: {elapsed:>7.1f}s")

    return best.sequence, best.objective, time.time()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    for inst in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, inst)
        if not os.path.exists(filepath):
            print(f"[SKIP] {inst} — archivo no encontrado")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)
        sheet_name = inst.replace(".txt", "")

        print(f"\n{'─'*60}")
        print(f"[BKGA] Procesando instancia: {inst}  (n={n}, m={m})")
        print(f"{'─'*60}")

        offsets_list = precompute_offsets(jobs)

        # Timer inicia ANTES del algoritmo
        t0 = time.time()

        # Ejecutar BKGA
        seq_bkga, z_bkga, time_finish = bkga(
            jobs, m, offsets_list, t0,
            pop_size=POP_SIZE,
            elite_size=ELITE_SIZE,
            mut_prob=MUT_PROB,
            tournament_k=TOURNAMENT_K,
            max_time=MAX_TIME,
        )

        compute_time_ms = round((time_finish - t0) * 1000)
        print(f"\n  Resultado final → Flujo total: {z_bkga:,} | Tiempo: {compute_time_ms/1000:.1f}s")

        # Obtener tiempos de inicio de cada job (operación 0) para el Excel
        _, schedule = evaluate_sequence_preciso(
            seq_bkga, jobs, m, offsets_list, save_schedule=True
        )
        job_start_times = [None] * n
        for op in schedule:
            if op["operation"] == 0:
                job_start_times[op["job"]] = op["start"]

        single_results = {sheet_name: (z_bkga, compute_time_ms, job_start_times)}
        write_results_to_excel(single_results, OUTPUT_FILE_BKGA)


if __name__ == "__main__":
    main()
