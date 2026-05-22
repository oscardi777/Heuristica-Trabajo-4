"""
compare.py
==========
Compara los resultados de los algoritmos EPR(VND) y BRKGA contra las cotas
inferiores (lower bounds) definidas en lb.txt, calculando el GAP para cada
instancia y generando una tabla resumen.

Uso:
    python compare.py

Archivos esperados:
    ./resultados/NWJSSP_OADG_EPR(VND).xlsx
    ./resultados/NWJSSP_OADG_BRKGA.xlsx
    ./lb.txt
"""

from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
RESULTS_DIR     = BASE_DIR / "resultados"
LB_FILE         = BASE_DIR / "lb.txt"
EPR_FILE        = RESULTS_DIR / "NWJSSP_OADG_EPR(VND).xlsx"
BRKGA_FILE      = RESULTS_DIR / "NWJSSP_OADG_BRKGA.xlsx"
OUTPUT_CSV      = RESULTS_DIR / "comparison_results.csv"


# ─────────────────────────────────────────────
# Lectura de cotas inferiores
# ─────────────────────────────────────────────
def read_lower_bounds(path: Path) -> list[int]:
    """
    Lee el archivo lb.txt y retorna una lista de enteros,
    uno por línea, en el mismo orden que las instancias.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de cotas inferiores: {path}")

    bounds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                bounds.append(int(line))
    return bounds


# ─────────────────────────────────────────────
# Lectura de resultados desde Excel
# ─────────────────────────────────────────────
def read_excel_results(path: Path) -> dict[str, float]:
    """
    Lee un archivo Excel de resultados.
    Cada hoja corresponde a una instancia.
    El valor Z (flujo total) está en la celda [fila 0, columna 0] de cada hoja.

    Retorna un dict {nombre_hoja: Z_value}.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo Excel: {path}")

    xl = pd.ExcelFile(path, engine="openpyxl")
    results = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        # Z está en la primera fila, primera columna
        z_value = df.iloc[0, 0]
        results[sheet] = float(z_value)
    return results


# ─────────────────────────────────────────────
# Cálculo del GAP
# ─────────────────────────────────────────────
def compute_gap(z: float, lb: int) -> float:
    """
    Calcula el GAP relativo entre el valor obtenido Z y la cota inferior LB.

    GAP = (Z - LB) / LB
    """
    if lb == 0:
        return float("inf")
    return (z - lb) / lb


# ─────────────────────────────────────────────
# Construcción de la tabla comparativa
# ─────────────────────────────────────────────
def build_comparison_table(
    epr_results: dict[str, float],
    brkga_results: dict[str, float],
    lower_bounds: list[int],
) -> pd.DataFrame:
    """
    Construye un DataFrame con columnas:
        Instancia | EPR(VND)_GAP | BRKGA_GAP

    Itera en orden sobre las hojas comunes a ambos Excels y las LBs disponibles.
    Es robusto: si una instancia falta en algún Excel, muestra NaN para ese método.
    """
    # Obtener las hojas que aparecen en al menos uno de los dos archivos
    all_sheets = sorted(
        set(epr_results.keys()) | set(brkga_results.keys()),
        key=lambda s: list(epr_results.keys()).index(s)
        if s in epr_results
        else list(brkga_results.keys()).index(s),
    )

    rows = []
    for idx, sheet in enumerate(all_sheets):
        if idx >= len(lower_bounds):
            print(f"[ADVERTENCIA] No hay cota inferior para la instancia '{sheet}' (índice {idx}). Se omite.")
            continue

        lb = lower_bounds[idx]
        z_epr   = epr_results.get(sheet, None)
        z_brkga = brkga_results.get(sheet, None)

        gap_epr   = compute_gap(z_epr, lb)   if z_epr   is not None else None
        gap_brkga = compute_gap(z_brkga, lb) if z_brkga is not None else None

        rows.append({
            "Instancia":    idx + 1,
            "EPR(VND)_GAP": round(gap_epr,   3) if gap_epr   is not None else None,
            "BRKGA_GAP":    round(gap_brkga, 3) if gap_brkga is not None else None,
        })

    df = pd.DataFrame(rows)

    # Fila de promedios
    avg_epr   = df["EPR(VND)_GAP"].mean()
    avg_brkga = df["BRKGA_GAP"].mean()
    avg_row = pd.DataFrame([{
        "Instancia":    "GAP promedio",
        "EPR(VND)_GAP": round(avg_epr,   3),
        "BRKGA_GAP":    round(avg_brkga, 3),
    }])
    df = pd.concat([df, avg_row], ignore_index=True)

    return df


# ─────────────────────────────────────────────
# Impresión en consola
# ─────────────────────────────────────────────
def print_table(df: pd.DataFrame) -> None:
    """
    Imprime la tabla de comparación en consola con formato legible.
    La última fila (GAP promedio) se resalta visualmente.
    """
    col_widths = {
        "Instancia":    14,
        "EPR(VND)_GAP": 14,
        "BRKGA_GAP":    12,
    }

    # Encabezado
    header = (
        f"{'Instancia':>{col_widths['Instancia']}} | "
        f"{'EPR(VND)_GAP':>{col_widths['EPR(VND)_GAP']}} | "
        f"{'BRKGA_GAP':>{col_widths['BRKGA_GAP']}}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for _, row in df.iterrows():
        inst      = str(row["Instancia"])
        gap_epr   = f"{row['EPR(VND)_GAP']:.3f}" if pd.notna(row["EPR(VND)_GAP"]) else "N/A"
        gap_brkga = f"{row['BRKGA_GAP']:.3f}"    if pd.notna(row["BRKGA_GAP"])    else "N/A"

        # Resaltar la fila de promedios
        prefix = "**" if inst == "GAP promedio" else "  "
        print(
            f"{prefix}{inst:>{col_widths['Instancia'] - 2}} | "
            f"{gap_epr:>{col_widths['EPR(VND)_GAP']}} | "
            f"{gap_brkga:>{col_widths['BRKGA_GAP']}}"
        )

    print(separator + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    print("=" * 50)
    print("  Comparación de algoritmos NWJSSP")
    print("=" * 50)

    # 1. Leer cotas inferiores
    print(f"\nLeyendo cotas inferiores desde: {LB_FILE}")
    lower_bounds = read_lower_bounds(LB_FILE)
    print(f"  → {len(lower_bounds)} cotas cargadas.")

    # 2. Leer resultados de ambos algoritmos
    print(f"\nLeyendo resultados EPR(VND) desde: {EPR_FILE}")
    epr_results = read_excel_results(EPR_FILE)
    print(f"  → {len(epr_results)} instancias encontradas: {list(epr_results.keys())}")

    print(f"\nLeyendo resultados BRKGA desde: {BRKGA_FILE}")
    brkga_results = read_excel_results(BRKGA_FILE)
    print(f"  → {len(brkga_results)} instancias encontradas: {list(brkga_results.keys())}")

    # 3. Construir tabla
    df = build_comparison_table(epr_results, brkga_results, lower_bounds)

    # 4. Mostrar en consola
    print_table(df)

    # 5. Guardar CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Tabla guardada en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
