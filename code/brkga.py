import os
import math
import time
import random
import pandas as pd

# ─────────────────────────────────────────────
# Archivos
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE_BRKGA = "resultados\\NWJSSP_OADG_BRKGA.xlsx"

# ─────────────────────────────────────────────
# Parámetros del Algoritmo Genético
# ─────────────────────────────────────────────
POP_SIZE       = 200    # Tamaño de la población
ELITE_SIZE     = 20     # Número de individuos élite que sobreviven cada generación
MUT_PROB       = 0.15   # Probabilidad de mutación por individuo
TOURNAMENT_K   = 2      # Tamaño del torneo para selección de padres
MAX_TIME       = 3600   # Tiempo máximo por instancia (segundos)
MAX_GEN_NO_IMPROVE = 300 # Número máximo de generaciones sin mejora antes de detener (opcional)
SIZE_MUTATE_KEYS = 0.20 # Porcentaje de genes a mutar (entre 10-20% recomendado)
BIAS_FATHER = 0.5         # Probabilidad de tomar el gen del padre 1 en el cruce uniforme

# ─────────────────────────────────────────────
# Instancias a procesar
# ─────────────────────────────────────────────
INSTANCES = [
    "ft06.txt",           "ft06r.txt",
    "ft10.txt",           "ft10r.txt",
    "ft20.txt",           "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    "tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
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
# CRUCE: SESGADO SOBRE RANDOM KEYS
# ─────────────────────────────────────────────
def bias_crossover(parent1: Individual, parent2: Individual, bias_father: float = BIAS_FATHER) -> Individual:
    """
    Cruce con sesgo: para cada posición, toma el valor del padre 1 o padre 2
    con probabilidad `BIAS_FATHER` y `1 - BIAS_FATHER` respectivamente.
    Retorna un nuevo individuo hijo (offspring).
    """
    n = len(parent1.keys)
    child_keys = [
        parent1.keys[i] if random.random() < bias_father else parent2.keys[i]
        for i in range(n)
    ]
    return Individual(child_keys)


# ─────────────────────────────────────────────
# MUTACIÓN: alteracion de los random keys
# ─────────────────────────────────────────────
def mutate(individual: Individual, mut_prob: float = 0.20) -> Individual:
    """
    Mutación Gaussiana para Random Keys.
    Mutamos un porcentaje de los genes con ruido gaussiano.
    """
    if random.random() >= mut_prob:
        return individual.copy()
    
    child = individual.copy()
    n = len(child.keys)
    
    # Garantizamos mutar al menos 1 gen
    num_mutations = max(1, int(SIZE_MUTATE_KEYS * n))
    
    for _ in range(num_mutations):
        i = random.randint(0, n - 1)
        # Perturbación gaussiana
        child.keys[i] += random.gauss(0, 0.15)
        # Mantener en [0, 1]
        child.keys[i] = max(0.0, min(1.0, child.keys[i]))
    
    child.sequence = sorted(range(n), key=lambda i: child.keys[i])
    
    child.objective = None
    
    return child


# ─────────────────────────────────────────────
# ALGORITMO GENÉTICO CON RANDOM KEYS (BRKGA)
# ─────────────────────────────────────────────
def brkga(jobs, m, offsets_list, start_time,
         pop_size=POP_SIZE, elite_size=ELITE_SIZE,
         mut_prob=MUT_PROB, tournament_k=TOURNAMENT_K,
         max_time=MAX_TIME, max_gen_no_improve=MAX_GEN_NO_IMPROVE, bias_father=BIAS_FATHER):
    
    n = len(jobs)
    generation = 0
    generations_no_improve = 0
    best_objective = float('inf')

    # Inicialización
    population = initialize_population(pop_size, n)
    for ind in population:
        ind.evaluate(jobs, m, offsets_list)
    population.sort(key=lambda x: x.objective)

    best = population[0].copy()
    best_objective = best.objective

    print(f"  Gen {generation:>4} | Mejor objetivo: {best.objective:>12,} | Tiempo: {time.time() - start_time:>7.1f}s")

    while (time.time() - start_time < max_time):
        generation += 1

        # Elite
        elite = [ind.copy() for ind in population[:elite_size]]

        # Generar offspring
        offspring = []
        offspring_size = pop_size - elite_size
        for _ in range(offspring_size):
            p1 = tournament_selection(population, tournament_k)
            p2 = tournament_selection(population, tournament_k)
            child = bias_crossover(p1, p2, bias_father)
            child = mutate(child, mut_prob)
            offspring.append(child)

        # Evaluar offspring
        for ind in offspring:
            ind.evaluate(jobs, m, offsets_list)

        # Nueva población
        population = elite + offspring
        population.sort(key=lambda x: x.objective)

        # === ACTUALIZACIÓN DEL MEJOR Y CONTADOR ===
        improved = False
        if population[0].objective < best.objective:
            best = population[0].copy()
            best_objective = best.objective
            generations_no_improve = 0
            improved = True
            print(f"  Gen {generation:>4} | Mejor objetivo: {best.objective:>12,} ★ | Tiempo: {time.time() - start_time:>7.1f}s")
        else:
            generations_no_improve += 1
            if generation % 50 == 0:
                print(f"  Gen {generation:>4} | Mejor objetivo: {best.objective:>12,}   | Tiempo: {time.time() - start_time:>7.1f}s")

        # Criterio de parada por estancamiento
        if generations_no_improve >= max_gen_no_improve:
            print(f"  → Parada por estancamiento ({max_gen_no_improve} generaciones sin mejora) en generación {generation}")
            break

    total_time = time.time() - start_time
    print(f"\nResultado final → Flujo total: {best.objective:,.0f} | Tiempo: {total_time:.1f}s")
    return best.sequence, best.objective, total_time


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
        print(f"[BRKGA] Procesando instancia: {inst}  (n={n}, m={m})")
        print(f"{'─'*60}")

        offsets_list = precompute_offsets(jobs)

        # Timer inicia ANTES del algoritmo
        t0 = time.time()

        # Ejecutar BRKGA
        seq_brkga, z_brkga, elapsed_time = brkga(
            jobs, m, offsets_list, t0,
            pop_size=POP_SIZE,
            elite_size=ELITE_SIZE,
            mut_prob=MUT_PROB,
            tournament_k=TOURNAMENT_K,
            max_time=MAX_TIME,
            max_gen_no_improve=MAX_GEN_NO_IMPROVE,
            bias_father=BIAS_FATHER
        )

        print(f"\n  Resultado final → Flujo total: {z_brkga:,} | Tiempo: {elapsed_time:.1f}s")

        # Obtener tiempos de inicio de cada job (operación 0) para el Excel
        _, schedule = evaluate_sequence_preciso(
            seq_brkga, jobs, m, offsets_list, save_schedule=True
        )
        job_start_times = [None] * n
        for op in schedule:
            if op["operation"] == 0:
                job_start_times[op["job"]] = op["start"]

        single_results = {sheet_name: (z_brkga, elapsed_time, job_start_times)}
        write_results_to_excel(single_results, OUTPUT_FILE_BRKGA)


if __name__ == "__main__":
    main()
