"""
experiments.py
==============
Ejecuta experimentos sistemáticos variando parámetros de EPR(VND) y BRKGA
sobre instancias del problema NWJSSP, calculando el GAP respecto a las
cotas inferiores de lb.txt.

Estructura de carpetas recomendada
-----------------------------------
proyecto/
├── code/
│   ├── epr.py
│   └── brkga.py
├── experiments/          ← tablas CSV generadas aquí (se crea automáticamente)
├── resultados/           ← Excels de resultados base
├── NWJSSP Instances/     ← archivos .txt de instancias
├── lb.txt
├── compare.py
└── experiments.py        ← ESTE ARCHIVO

Uso:
    python experiments.py

El script importa las funciones de epr.py y brkga.py desde ./code/.
"""

import sys
import time
import random
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────
# Agregar ./code al path para importar los módulos
# ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
CODE_DIR  = BASE_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

# Importar funciones compartidas de los algoritmos
# (ambos módulos exponen las mismas utilidades de lectura/evaluación)
from epr import (
    read_instance,
    precompute_offsets,
    evaluate_sequence_preciso,
    epr,
)
from brkga import brkga

# ─────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────
INSTANCES_DIR  = BASE_DIR / "NWJSSP Instances"
LB_FILE        = BASE_DIR / "lb.txt"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# ─────────────────────────────────────────────
# Instancias de experimento (subconjunto pequeño
# para que los experimentos sean manejables en tiempo)
# ─────────────────────────────────────────────
EXPERIMENT_INSTANCES = [
    "ft06.txt",           "ft06r.txt",
    "ft10.txt",           "ft10r.txt",
    "ft20.txt",           "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
]

# ─────────────────────────────────────────────
# Parámetros base (valores originales de cada algoritmo)
# ─────────────────────────────────────────────
EPR_BASE_PARAMS = {
    "pop_size":            100,
    "elite_size":          25,
    "mut_prob":            0.20,
    "bias_father":         0.75,
    "tournament_k":        2,
    "max_time":            3600,
    "max_gen_no_improve":  150,
    "epr_freq":            25,
    "elite_pool_size":     5,
}

BRKGA_BASE_PARAMS = {
    "pop_size":            100,
    "elite_size":          25,
    "mut_prob":            0.20,
    "bias_father":         0.75,
    "tournament_k":        2,
    "max_time":            3600,
    "max_gen_no_improve":  250,
}

# Tiempo máximo reducido para experimentos (evitar timeouts largos)
EXPERIMENT_MAX_TIME = 120   # segundos por ejecución de experimento

# ─────────────────────────────────────────────
# Valores a variar por parámetro (EPR)
# ─────────────────────────────────────────────
EPR_VARIATIONS = {
    "pop_size":           [50, 100, 200],
    "elite_size":         [10, 25, 40],
    "mut_prob":           [0.10, 0.20, 0.40],
    "bias_father":        [0.50, 0.75, 0.90],
    "epr_freq":           [10, 25, 50],
    "elite_pool_size":    [3, 5, 10],
    "max_gen_no_improve": [50, 150, 300],
}

# ─────────────────────────────────────────────
# Valores a variar por parámetro (BRKGA)
# ─────────────────────────────────────────────
BRKGA_VARIATIONS = {
    "pop_size":           [50, 100, 200],
    "elite_size":         [10, 25, 40],
    "mut_prob":           [0.10, 0.20, 0.40],
    "bias_father":        [0.50, 0.75, 0.90],
    "max_gen_no_improve": [50, 150, 300],
}


# ═════════════════════════════════════════════
# UTILIDADES GENERALES
# ═════════════════════════════════════════════

def read_lower_bounds(path: Path) -> list[int]:
    """
    Lee lb.txt y retorna lista de cotas inferiores (enteros).
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró lb.txt en: {path}")
    bounds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                bounds.append(int(line))
    return bounds


def compute_gap(z: float, lb: int) -> float:
    """GAP = (Z - LB) / LB"""
    return (z - lb) / lb if lb != 0 else float("inf")


def load_instances(instance_names: list[str]) -> list[tuple[str, list, int, list, int]]:
    """
    Carga las instancias especificadas.
    Retorna lista de tuplas: (nombre, jobs, m, offsets_list, lb_index).
    Solo carga las que existen en disco.
    """
    loaded = []
    lb_index = 0  # Índice en lb.txt (se incrementa por cada instancia procesada)

    # Leer el mapping de nombre → índice LB (por orden de aparición en EXPERIMENT_INSTANCES)
    for name in instance_names:
        path = INSTANCES_DIR / name
        if not path.exists():
            print(f"  [SKIP] {name} — no encontrado en {INSTANCES_DIR}")
            lb_index += 1
            continue
        jobs, m = read_instance(str(path))
        offsets  = precompute_offsets(jobs)
        loaded.append((name, jobs, m, offsets, lb_index))
        lb_index += 1

    return loaded


# ═════════════════════════════════════════════
# RUNNER DE EXPERIMENTO
# ═════════════════════════════════════════════

def run_epr_experiment(
    instances: list,
    lower_bounds: list[int],
    params: dict,
) -> dict[str, float]:
    """
    Ejecuta EPR sobre las instancias dadas con los parámetros indicados.
    Retorna dict {nombre_instancia: gap}.
    """
    results = {}
    for name, jobs, m, offsets, lb_idx in instances:
        if lb_idx >= len(lower_bounds):
            print(f"  [ADVERTENCIA] Sin LB para instancia '{name}'.")
            continue

        lb = lower_bounds[lb_idx]
        random.seed(42)  # Reproducibilidad

        t0 = time.time()
        # Usar el tiempo máximo reducido para experimentos
        p = {**params, "max_time": EXPERIMENT_MAX_TIME}

        seq, z, _ = epr(
            jobs, m, offsets, t0,
            pop_size           = p["pop_size"],
            elite_size         = p["elite_size"],
            mut_prob           = p["mut_prob"],
            bias_father        = p["bias_father"],
            tournament_k       = p["tournament_k"],
            max_time           = p["max_time"],
            max_gen_no_improve = p["max_gen_no_improve"],
            epr_freq           = p["epr_freq"],
            elite_pool_size    = p["elite_pool_size"],
        )
        gap = compute_gap(z, lb)
        results[name] = round(gap, 3)
        print(f"    {name:35s} → Z={z:>12,}  LB={lb:>12,}  GAP={gap:.3f}")

    return results


def run_brkga_experiment(
    instances: list,
    lower_bounds: list[int],
    params: dict,
) -> dict[str, float]:
    """
    Ejecuta BRKGA sobre las instancias dadas con los parámetros indicados.
    Retorna dict {nombre_instancia: gap}.
    """
    results = {}
    for name, jobs, m, offsets, lb_idx in instances:
        if lb_idx >= len(lower_bounds):
            print(f"  [ADVERTENCIA] Sin LB para instancia '{name}'.")
            continue

        lb = lower_bounds[lb_idx]
        random.seed(42)

        t0 = time.time()
        p  = {**params, "max_time": EXPERIMENT_MAX_TIME}

        seq, z, _ = brkga(
            jobs, m, offsets, t0,
            pop_size           = p["pop_size"],
            elite_size         = p["elite_size"],
            mut_prob           = p["mut_prob"],
            bias_father        = p["bias_father"],
            tournament_k       = p["tournament_k"],
            max_time           = p["max_time"],
            max_gen_no_improve = p["max_gen_no_improve"],
        )
        gap = compute_gap(z, lb)
        results[name] = round(gap, 3)
        print(f"    {name:35s} → Z={z:>12,}  LB={lb:>12,}  GAP={gap:.3f}")

    return results


# ═════════════════════════════════════════════
# CONSTRUCCIÓN DE TABLA DE EXPERIMENTO
# ═════════════════════════════════════════════

def build_experiment_table(
    param_name: str,
    param_values: list,
    all_results: list[dict[str, float]],
    instance_names: list[str],
) -> pd.DataFrame:
    """
    Construye un DataFrame con columnas:
        <param_name> | GAP_inst1 | GAP_inst2 | ... | GAP_promedio

    Parámetros:
        param_name    : nombre del parámetro variado
        param_values  : lista de valores probados
        all_results   : lista de dicts {inst_name: gap} (uno por valor de param)
        instance_names: nombres cortos de las instancias (sin .txt)
    """
    rows = []
    for val, gaps in zip(param_values, all_results):
        row = {param_name: val}
        gap_vals = []
        for inst in instance_names:
            g = gaps.get(inst, None)
            row[inst.replace(".txt", "")] = g
            if g is not None:
                gap_vals.append(g)
        row["GAP_promedio"] = round(sum(gap_vals) / len(gap_vals), 3) if gap_vals else None
        rows.append(row)

    return pd.DataFrame(rows)


def save_experiment_table(df: pd.DataFrame, filename: str) -> None:
    """Guarda un DataFrame en la carpeta experiments/ como CSV."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"  → Guardado: {out_path}")


def print_experiment_table(param_name: str, df: pd.DataFrame) -> None:
    """Imprime una tabla de experimento en consola de forma legible."""
    print(f"\n  Variando '{param_name}':")
    print(df.to_string(index=False))


# ═════════════════════════════════════════════
# EXPERIMENTOS EPR
# ═════════════════════════════════════════════

def run_epr_experiments(instances: list, lower_bounds: list[int]) -> None:
    """
    Ejecuta todos los experimentos de EPR variando un parámetro por vez.
    Para cada parámetro, prueba los valores definidos en EPR_VARIATIONS,
    manteniendo el resto en sus valores base (EPR_BASE_PARAMS).
    Guarda una tabla CSV por parámetro variado.
    """
    instance_names = [name for name, *_ in instances]
    summary = {}

    print("\n" + "═" * 60)
    print("  EXPERIMENTOS EPR(VND)")
    print("═" * 60)

    for param_name, param_values in EPR_VARIATIONS.items():
        print(f"\n  ── Variando {param_name}: {param_values}")
        all_results = []

        for val in param_values:
            # Parámetros base + valor variado
            params = {**EPR_BASE_PARAMS, param_name: val}
            print(f"\n    [{param_name} = {val}]")
            gaps = run_epr_experiment(instances, lower_bounds, params)
            all_results.append(gaps)

        # Construir y guardar tabla
        df = build_experiment_table(param_name, param_values, all_results, instance_names)
        filename = f"epr_varying_{param_name}.csv"
        save_experiment_table(df, filename)
        print_experiment_table(param_name, df)

        # Guardar promedio para el resumen final
        summary[f"EPR / {param_name}"] = {
            str(v): round(res.get(inst, float("nan")), 3)
            for v, res in zip(param_values, all_results)
            for inst in instance_names[:1]  # Solo primera instancia en resumen
        }

    return summary


# ═════════════════════════════════════════════
# EXPERIMENTOS BRKGA
# ═════════════════════════════════════════════

def run_brkga_experiments(instances: list, lower_bounds: list[int]) -> None:
    """
    Ejecuta todos los experimentos de BRKGA variando un parámetro por vez.
    Para cada parámetro, prueba los valores definidos en BRKGA_VARIATIONS,
    manteniendo el resto en sus valores base (BRKGA_BASE_PARAMS).
    Guarda una tabla CSV por parámetro variado.
    """
    instance_names = [name for name, *_ in instances]
    summary = {}

    print("\n" + "═" * 60)
    print("  EXPERIMENTOS BRKGA")
    print("═" * 60)

    for param_name, param_values in BRKGA_VARIATIONS.items():
        print(f"\n  ── Variando {param_name}: {param_values}")
        all_results = []

        for val in param_values:
            params = {**BRKGA_BASE_PARAMS, param_name: val}
            print(f"\n    [{param_name} = {val}]")
            gaps = run_brkga_experiment(instances, lower_bounds, params)
            all_results.append(gaps)

        # Construir y guardar tabla
        df = build_experiment_table(param_name, param_values, all_results, instance_names)
        filename = f"brkga_varying_{param_name}.csv"
        save_experiment_table(df, filename)
        print_experiment_table(param_name, df)

        summary[f"BRKGA / {param_name}"] = {
            str(v): round(res.get(inst, float("nan")), 3)
            for v, res in zip(param_values, all_results)
            for inst in instance_names[:1]
        }

    return summary


# ═════════════════════════════════════════════
# RESUMEN FINAL
# ═════════════════════════════════════════════

def print_final_summary(epr_summary: dict, brkga_summary: dict) -> None:
    """Imprime un resumen ejecutivo de los experimentos completados."""
    print("\n" + "═" * 60)
    print("  RESUMEN DE EXPERIMENTOS COMPLETADOS")
    print("═" * 60)

    print("\n  Archivos CSV generados en ./experiments/:")
    for f in sorted(EXPERIMENTS_DIR.glob("*.csv")):
        print(f"    • {f.name}")

    print(f"\n  Total experimentos EPR   : {len(EPR_VARIATIONS)} parámetros × {sum(len(v) for v in EPR_VARIATIONS.values())} configuraciones")
    print(f"  Total experimentos BRKGA : {len(BRKGA_VARIATIONS)} parámetros × {sum(len(v) for v in BRKGA_VARIATIONS.values())} configuraciones")
    print()


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Experimentos sistemáticos — NWJSSP")
    print("=" * 60)

    # 1. Leer cotas inferiores
    print(f"\nLeyendo cotas inferiores: {LB_FILE}")
    lower_bounds = read_lower_bounds(LB_FILE)
    print(f"  → {len(lower_bounds)} cotas cargadas.")

    # 2. Cargar instancias
    print(f"\nCargando instancias desde: {INSTANCES_DIR}")
    instances = load_instances(EXPERIMENT_INSTANCES)
    print(f"  → {len(instances)} instancias cargadas.")

    if not instances:
        print("[ERROR] No se encontraron instancias. Verifica INSTANCES_DIR.")
        return

    # 3. Ejecutar experimentos EPR
    epr_summary = run_epr_experiments(instances, lower_bounds)

    # 4. Ejecutar experimentos BRKGA
    brkga_summary = run_brkga_experiments(instances, lower_bounds)

    # 5. Resumen final
    print_final_summary(epr_summary, brkga_summary)


if __name__ == "__main__":
    main()
