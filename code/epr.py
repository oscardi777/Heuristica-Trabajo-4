import os
import math
import time
import random
import pandas as pd

# ─────────────────────────────────────────────
# Archivos
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE_EPR = "resultados\\NWJSSP_OADG_EPR(VND).xlsx"

# ─────────────────────────────────────────────
# Parámetros
# ─────────────────────────────────────────────
POP_SIZE = 250
ELITE_SIZE = 25
MUT_PROB = 0.20
BIAS_FATHER = 0.75
MAX_GEN_NO_IMPROVE = 250
SIZE_MUTATE_KEYS = 0.15
TOURNAMENT_K = 2
MAX_TIME = 3600

# EPR
PR_INTERVAL = 10
MAX_BLOCK_SIZE = 5

random.seed(42)

# ─────────────────────────────────────────────
# Instancias
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

# ─────────────────────────────────────────────
# ESTRUCTURAS
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
        self.intervals = []

    def add(self, b: int, e: int):
        self.intervals.append((b, e))

    def max_end_before(self, threshold: int):
        max_e = 0
        for b, e in self.intervals:
            if b < threshold:
                max_e = max(max_e, e)
        return max_e


# ─────────────────────────────────────────────
# LECTURA
# ─────────────────────────────────────────────
def read_instance(file):
    with open(file) as f:
        n, m = map(int, f.readline().split())

        jobs = []

        for _ in range(n):
            data = list(map(int, f.readline().split()))

            operations = [
                Operation(data[2*i], data[2*i + 1])
                for i in range(m)
            ]

            jobs.append(Job(operations, release=data[-1]))

    return jobs, m


# ─────────────────────────────────────────────
# OFFSETS
# ─────────────────────────────────────────────
def precompute_offsets(jobs):

    return [
        [0] * len(job.operations)
        if len(job.operations) <= 1
        else
        [0] + [
            sum(op.p for op in job.operations[:u+1])
            for u in range(len(job.operations)-1)
        ]
        for job in jobs
    ]


# ─────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────
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

        total_flow += schedule_job_preciso(
            jobs[j],
            machines,
            j,
            schedule,
            offsets_list[j]
        )

    return (total_flow, schedule) if save_schedule else total_flow


# ─────────────────────────────────────────────
# INDIVIDUOS
# ─────────────────────────────────────────────
class Individual:

    def __init__(self, keys):

        self.keys = keys

        self.sequence = sorted(
            range(len(keys)),
            key=lambda i: keys[i]
        )

        self.objective = None

    def evaluate(self, jobs, m, offsets_list):

        if self.objective is None:

            self.objective = evaluate_sequence_preciso(
                self.sequence,
                jobs,
                m,
                offsets_list
            )

        return self.objective

    def copy(self):

        ind = Individual(self.keys[:])

        ind.objective = self.objective

        return ind


# ─────────────────────────────────────────────
# POBLACIÓN
# ─────────────────────────────────────────────
def initialize_population(pop_size, n):

    return [
        Individual([random.random() for _ in range(n)])
        for _ in range(pop_size)
    ]


# ─────────────────────────────────────────────
# SELECCIÓN
# ─────────────────────────────────────────────
def tournament_selection(population, k=TOURNAMENT_K):

    candidates = random.sample(population, k)

    return min(candidates, key=lambda ind: ind.objective)


# ─────────────────────────────────────────────
# CRUCE
# ─────────────────────────────────────────────
def bias_crossover(parent1, parent2, bias_father=BIAS_FATHER):

    if parent2.objective < parent1.objective:
        parent1, parent2 = parent2, parent1

    n = len(parent1.keys)

    child_keys = [

        parent1.keys[i]
        if random.random() < bias_father
        else parent2.keys[i]

        for i in range(n)
    ]

    return Individual(child_keys)


# ─────────────────────────────────────────────
# MUTACIÓN
# ─────────────────────────────────────────────
def mutate(individual, mut_prob=MUT_PROB):

    if random.random() >= mut_prob:
        return individual.copy()

    child = individual.copy()

    n = len(child.keys)

    num_mutations = max(1, int(SIZE_MUTATE_KEYS * n))

    for _ in range(num_mutations):

        i = random.randint(0, n - 1)

        child.keys[i] += random.gauss(0, 0.15)

        child.keys[i] = max(
            0.0,
            min(1.0, child.keys[i])
        )

    child.sequence = sorted(
        range(n),
        key=lambda i: child.keys[i]
    )

    child.objective = None

    return child


# ─────────────────────────────────────────────
# VECINDARIOS VND
# ─────────────────────────────────────────────
def generate_insertion_down_neighbors(sequence):

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

    neighbors = []

    n = len(sequence)

    for i in range(n):

        for j in range(i + 1, n):

            new_seq = sequence[:]

            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

            neighbors.append(new_seq)

    return neighbors


def generate_insertion_up_neighbors(sequence):

    neighbors = []

    n = len(sequence)

    for i in range(n):

        for j in range(i):

            new_seq = sequence[:]

            job = new_seq.pop(i)

            new_seq.insert(j, job)

            neighbors.append(new_seq)

    return neighbors


# ─────────────────────────────────────────────
# LOCAL SEARCH
# ─────────────────────────────────────────────
def local_search_first_improvement(
    initial_sequence,
    jobs,
    m,
    offsets_list,
    start_time,
    k,
    current_z,
    NEIGHBORS_GENERATOR
):

    B = list(initial_sequence)

    fin = False

    while not fin and (time.time() - start_time < MAX_TIME):

        fin = True

        neighbors = NEIGHBORS_GENERATOR[k](B)

        for P in neighbors:

            if time.time() - start_time >= MAX_TIME:
                break

            z = evaluate_sequence_preciso(
                P,
                jobs,
                m,
                offsets_list
            )

            if z < current_z:

                B = P
                current_z = z

                fin = False

                break

    return B, current_z, time.time()


# ─────────────────────────────────────────────
# VND
# ─────────────────────────────────────────────
def vnd(initial_sequence, jobs, m, offsets_list, start_time):

    B = list(initial_sequence)

    current_z = evaluate_sequence_preciso(
        B,
        jobs,
        m,
        offsets_list
    )

    NEIGHBORS_GENERATOR = [

        generate_insertion_down_neighbors,
        generate_swap_neighbors,
        generate_insertion_up_neighbors
    ]

    k = 0

    while (
        time.time() - start_time < MAX_TIME
        and
        k < len(NEIGHBORS_GENERATOR)
    ):

        candidate, z_candidate, _ = local_search_first_improvement(
            B,
            jobs,
            m,
            offsets_list,
            start_time,
            k,
            current_z,
            NEIGHBORS_GENERATOR
        )

        if z_candidate < current_z:

            k = 0

            B = candidate

            current_z = z_candidate

        else:

            k += 1

    return B, current_z, time.time()


# ─────────────────────────────────────────────
# PATH RELINKING
# ─────────────────────────────────────────────
def positions_map(sequence):

    return {
        job: idx
        for idx, job in enumerate(sequence)
    }


def apply_block_move(sequence, start, end, insert_pos):

    seq = sequence[:]

    block = seq[start:end]

    del seq[start:end]

    if insert_pos > start:
        insert_pos -= (end - start)

    for i, val in enumerate(block):
        seq.insert(insert_pos + i, val)

    return seq


def deterministic_block_relink(current, guide):

    current = current[:]

    n = len(current)

    for i in range(n):

        if current[i] != guide[i]:

            target_job = guide[i]

            j = current.index(target_job)

            end = j + 1

            while (
                end < n
                and
                end - j < MAX_BLOCK_SIZE
                and
                i + (end - j) < n
                and
                current[end] == guide[i + (end - j)]
            ):

                end += 1

            current = apply_block_move(
                current,
                j,
                end,
                i
            )

            return current

    return current


def path_relinking(source, target):

    path = []

    current = source[:]

    while current != target:

        current = deterministic_block_relink(
            current,
            target
        )

        path.append(current[:])

    return path


def select_path_solution(path, jobs, m, offsets_list):

    best_seq = None

    best_z = float("inf")

    for seq in path:

        z = evaluate_sequence_preciso(
            seq,
            jobs,
            m,
            offsets_list
        )

        if z < best_z:

            best_z = z
            best_seq = seq

    return best_seq, best_z


# ─────────────────────────────────────────────
# SEQUENCE → INDIVIDUAL
# ─────────────────────────────────────────────
def sequence_to_individual(sequence):

    n = len(sequence)

    keys = [0.0] * n

    for rank, job in enumerate(sequence):

        keys[job] = rank / n

    return Individual(keys)


# ─────────────────────────────────────────────
# EPR
# ─────────────────────────────────────────────
def epr(
    jobs,
    m,
    offsets_list,
    start_time,

    pop_size=POP_SIZE,
    elite_size=ELITE_SIZE,
    mut_prob=MUT_PROB,
    tournament_k=TOURNAMENT_K,
    max_time=MAX_TIME,
    max_gen_no_improve=MAX_GEN_NO_IMPROVE,
    bias_father=BIAS_FATHER
):

    n = len(jobs)

    generation = 0

    generations_no_improve = 0

    # Inicialización
    population = initialize_population(pop_size, n)

    for ind in population:
        ind.evaluate(jobs, m, offsets_list)

    population.sort(key=lambda x: x.objective)

    best = population[0].copy()

    print(
        f"  Gen {generation:>4} | "
        f"Best: {best.objective:>12,}"
    )

    while time.time() - start_time < max_time:

        generation += 1

        # ─────────────────────────
        # ELITE
        # ─────────────────────────
        elite = [
            ind.copy()
            for ind in population[:elite_size]
        ]

        # ─────────────────────────
        # OFFSPRING BRKGA
        # ─────────────────────────
        offspring = []

        offspring_size = pop_size - elite_size

        for _ in range(offspring_size):

            p1 = tournament_selection(population, tournament_k)

            p2 = tournament_selection(population, tournament_k)

            child = bias_crossover(
                p1,
                p2,
                bias_father
            )

            child = mutate(child, mut_prob)

            offspring.append(child)

        for ind in offspring:
            ind.evaluate(jobs, m, offsets_list)

        population = elite + offspring

        # ─────────────────────────
        # EPR
        # ─────────────────────────
        if generation % PR_INTERVAL == 0:

            S = []

            for i in range(elite_size):

                Pi = elite[i]

                j = random.randint(0, elite_size - 1)

                while j == i:
                    j = random.randint(0, elite_size - 1)

                Pj = elite[j]

                path = path_relinking(
                    Pi.sequence,
                    Pj.sequence
                )

                if len(path) == 0:
                    continue

                seq_pr, z_pr = select_path_solution(
                    path,
                    jobs,
                    m,
                    offsets_list
                )

                seq_vnd, z_vnd, _ = vnd(
                    seq_pr,
                    jobs,
                    m,
                    offsets_list,
                    start_time
                )

                ind = sequence_to_individual(seq_vnd)

                ind.objective = z_vnd

                S.append(ind)

            population.extend(S)

        # ─────────────────────────
        # NUEVA POBLACIÓN
        # ─────────────────────────
        population.sort(key=lambda x: x.objective)

        population = population[:pop_size]

        # ─────────────────────────
        # MEJOR
        # ─────────────────────────
        if population[0].objective < best.objective:

            best = population[0].copy()

            generations_no_improve = 0

            print(
                f"  Gen {generation:>4} | "
                f"Best: {best.objective:>12,} ★"
            )

        else:

            generations_no_improve += 1

            if generation % 50 == 0:

                print(
                    f"  Gen {generation:>4} | "
                    f"Best: {best.objective:>12,}"
                )

        # ─────────────────────────
        # ESTANCAMIENTO
        # ─────────────────────────
        if generations_no_improve >= max_gen_no_improve:

            print(
                f"\n→ Parada por estancamiento "
                f"en generación {generation}"
            )

            break

    total_time = round(
        (time.time() - start_time) * 1000
    )

    print(
        f"\nResultado final → "
        f"Flowtime: {best.objective:,} | "
        f"Tiempo: {total_time:.1f}ms"
    )

    return best.sequence, best.objective, total_time


# ─────────────────────────────────────────────
# EXCEL
# ─────────────────────────────────────────────
def write_results_to_excel(results, output_file):

    os.makedirs(
        os.path.dirname(output_file)
        if os.path.dirname(output_file)
        else ".",
        exist_ok=True
    )

    writer_kwargs = (

        dict(
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace"
        )

        if os.path.exists(output_file)

        else

        dict(
            engine="openpyxl",
            mode="w"
        )
    )

    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:

        for sheet_name, (
            total_flow,
            compute_time_ms,
            job_start_times
        ) in results.items():

            df = pd.DataFrame([
                [total_flow, compute_time_ms],
                job_start_times
            ])

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                header=False,
                index=False
            )

    print(f"  → Resultados guardados en: {output_file}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    for inst in INSTANCES:

        filepath = os.path.join(
            INSTANCES_DIR,
            inst
        )

        if not os.path.exists(filepath):

            print(
                f"[SKIP] {inst} "
                f"— archivo no encontrado"
            )

            continue

        jobs, m = read_instance(filepath)

        n = len(jobs)

        sheet_name = inst.replace(".txt", "")

        print(f"\n{'─'*60}")
        print(f"[EPR] Procesando instancia: {inst}")
        print(f"{'─'*60}")

        offsets_list = precompute_offsets(jobs)

        t0 = time.time()

        # EPR
        seq_epr, z_epr, elapsed_time = epr(
            jobs,
            m,
            offsets_list,
            t0
        )

        # Schedule Excel
        _, schedule = evaluate_sequence_preciso(
            seq_epr,
            jobs,
            m,
            offsets_list,
            save_schedule=True
        )

        job_start_times = [None] * n

        for op in schedule:

            if op["operation"] == 0:

                job_start_times[op["job"]] = op["start"]

        single_results = {
            sheet_name: (
                z_epr,
                elapsed_time,
                job_start_times
            )
        }

        write_results_to_excel(
            single_results,
            OUTPUT_FILE_EPR
        )


if __name__ == "__main__":
    main()