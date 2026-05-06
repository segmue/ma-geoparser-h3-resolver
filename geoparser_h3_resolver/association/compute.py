"""
Spatial Association Analysis: NPMI + Gewichtungsmatrizen (B1, B2).

Berechnet fuer jede Kombination von OBJEKTARTs im Datensatz:
  - NPMI  (symmetrisch)  : Normalized Pointwise Mutual Information
  - B1    (asymmetrisch) : Kontextgewichtung   = NPMI * p_b / (p_a + p_b)
  - B2    (asymmetrisch) : Konfidenzgewichtung  = NPMI * p_ab / p_a

Alle Berechnungen sind flaechenbasiert (h3_cell_area)

Verwendung:
    from geoparser_h3_resolver.association import compute_all
    npmi, b1, b2 = compute_all("data/spatial_h3.duckdb")
"""

import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from h3_multi_resolution_engine import H3Engine


def calculate_npmi(p_a: float, p_b: float, p_ab: float) -> float:
    """Normalized Pointwise Mutual Information.

    NPMI = log2(p_ab / (p_a * p_b)) / (-log2(p_ab))

    Wertebereich: [-1, 1]
      -1 = nie zusammen
       0 = unabhaengig (Zufall)
      +1 = perfekte Co-Occurrence
    """
    eps = 1e-15
    if p_ab <= eps:
        return -1.0
    pmi = np.log2(p_ab / (p_a * p_b))
    norm_factor = -np.log2(p_ab)
    if norm_factor == 0:
        return 0.0
    result = pmi / norm_factor
    return float(max(min(result, 1.0), -1.0))


def compute_all(
    db_path: str | Path,
    total_area_resolution: int = 10,
    output_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Berechnet NPMI, B1 und B2 Matrizen fuer alle OBJEKTART-Paare.

    Args:
        db_path: Pfad zur DuckDB Datei
        total_area_resolution: Resolution fuer Gesamtflaechen-Vereinfachung
        output_dir: Optionaler Pfad zum Speichern der Matrizen als CSV

    Returns:
        Tuple von (npmi_df, b1_df, b2_df) als pandas DataFrames
    """
    engine = H3Engine(db_path)

    # 1. Alle OBJEKTARTs ermitteln
    objektarten = engine.conn.execute("""
        SELECT DISTINCT OBJEKTART
        FROM features
        ORDER BY OBJEKTART
    """).fetchall()
    objektarten = [row[0] for row in objektarten]
    n = len(objektarten)
    print(f"Gefunden: {n} OBJEKTARTs -> {n*(n-1)//2} Paare")

    # 2. Gesamtflaeche
    print(f"\nBerechne Gesamtflaeche (Resolution {total_area_resolution})...")
    t0 = time.time()
    total_area = engine.total_area(resolution=total_area_resolution)
    print(f"  Gesamtflaeche: {total_area:,.2f} km^2  ({time.time()-t0:.1f}s)")

    # 3. Pro OBJEKTART: Union CellSets und Flaechen
    print(f"\nBerechne Unions und Flaechen pro OBJEKTART ({n} Stueck)...")
    t0 = time.time()
    unions: dict[str, any] = {}
    areas: dict[str, float] = {}
    for obj in objektarten:
        unions[obj] = engine.union(f"OBJEKTART = '{obj}'")
        areas[obj] = engine.area(unions[obj])
    print(f"  Fertig ({time.time()-t0:.1f}s)")

    # 4. Paarweise Intersection-Flaechen
    pairs = list(combinations(objektarten, 2))
    n_pairs = len(pairs)
    print(f"\nBerechne Intersection-Flaechen ({n_pairs} Paare)...")

    intersection_areas: dict[tuple[str, str], float] = {}
    t0 = time.time()
    for i, (obj_a, obj_b) in enumerate(pairs):
        intersection_areas[(obj_a, obj_b)] = engine.area(
            engine.intersection(unions[obj_a], unions[obj_b])
        )

        if (i + 1) % 100 == 0 or (i + 1) == n_pairs:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_pairs - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{n_pairs} ({elapsed:.0f}s, ~{remaining:.0f}s verbleibend)")

    # 5. Matrizen aufbauen
    print("\nBerechne NPMI, B1, B2 Matrizen...")
    idx = {name: i for i, name in enumerate(objektarten)}

    npmi_matrix = np.full((n, n), np.nan)
    b1_matrix = np.full((n, n), np.nan)
    b2_matrix = np.full((n, n), np.nan)

    # Diagonale
    for obj in objektarten:
        i = idx[obj]
        p_a = areas[obj] / total_area
        npmi_matrix[i, i] = calculate_npmi(p_a, p_a, p_a)
        b1_matrix[i, i] = npmi_matrix[i, i] * 0.5
        b2_matrix[i, i] = npmi_matrix[i, i] * 1.0

    # Alle Paare
    for (obj_a, obj_b), area_ab in intersection_areas.items():
        i = idx[obj_a]
        j = idx[obj_b]

        p_a = areas[obj_a] / total_area
        p_b = areas[obj_b] / total_area
        p_ab = area_ab / total_area

        npmi_val = calculate_npmi(p_a, p_b, p_ab)

        npmi_matrix[i, j] = npmi_val
        npmi_matrix[j, i] = npmi_val

        denom = p_a + p_b
        if denom > 0:
            b1_matrix[i, j] = npmi_val * (p_b / denom)
            b1_matrix[j, i] = npmi_val * (p_a / denom)

        b2_matrix[i, j] = npmi_val * (p_ab / p_a) if p_a > 0 else 0.0
        b2_matrix[j, i] = npmi_val * (p_ab / p_b) if p_b > 0 else 0.0

    # 6. Als DataFrames
    npmi_df = pd.DataFrame(npmi_matrix, index=objektarten, columns=objektarten)
    b1_df = pd.DataFrame(b1_matrix, index=objektarten, columns=objektarten)
    b2_df = pd.DataFrame(b2_matrix, index=objektarten, columns=objektarten)

    # 7. Optional speichern
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        npmi_df.to_csv(out / "npmi_matrix.csv", sep=";")
        b1_df.to_csv(out / "b1_matrix.csv", sep=";")
        b2_df.to_csv(out / "b2_matrix.csv", sep=";")
        print(f"\nMatrizen gespeichert in {out}/")

    engine.close()

    print(f"\nFertig: {n}x{n} Matrizen berechnet.")
    return npmi_df, b1_df, b2_df
