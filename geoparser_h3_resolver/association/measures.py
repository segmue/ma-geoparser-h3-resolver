"""
Assoziationsmasse: Registry und die reinen Rechenbausteine.

Zwei Masse, ueber BuildConfig.association_measure waehlbar:

  "b1" (Default, bisheriges Verhalten) — das Ueberlagerungsmass
      Geschaetzt aus der Ueberlagerung der H3-Zellmengen: p_a ist der
      Flaechenanteil einer Kategorie, p_ab der Anteil der gemeinsamen Flaeche.
      Die Richtung wird nachtraeglich ueber einen Flaechenfaktor gesetzt
      (compute.py, Kapitel 4.2 Formeln 4.2/4.3 der Arbeit).

  "d1" — das Adjazenzmass
      Geschaetzt aus H3-Ringraengen: jedes Objekt verteilt sein Gewicht ueber
      seine eigenen Zellen und deren naechste Ringe. Die Richtung steckt hier
      schon in der Konstruktion (d1.py, Formeln 4.4/4.5).

Beide enden in derselben NPMI-Normierung (Formel 4.1, nach Bouma 2009) und in
demselben CSV-Format: quadratisch, `;`-separiert, OBJEKTART als Index und
Kopfzeile, Zeile = Quellkategorie. Der Resolver liest beide ueber denselben
AssociationMatrixLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

B1 = "b1"
D1 = "d1"
VALID_MEASURES = (B1, D1)


@dataclass
class B1Params:
    """Parameter des Ueberlagerungsmasses B1.

    Attributes:
        total_area_resolution: Resolution, auf der die Gesamtflaeche des
            Referenzgebiets bestimmt wird (Nenner von p_a, p_b und p_ab).
    """

    total_area_resolution: int = 10


@dataclass
class D1Params:
    """Parameter des Adjazenzmasses D1.

    Die ersten vier sind gesetzt und nicht aus den Daten abgeleitet; sie tragen
    die in der Arbeit berichteten Zahlen.

    Attributes:
        r_star: Ankeraufloesung, auf die alle Zellen abgebildet werden.
        cap: Hoechstzahl Stichprobenzellen je Objekt (gleichverteilte Auswahl
            ueber hash(cell)). Der Default 99999 ist der Wert des berichteten
            Laufs und deckelt faktisch nichts — das groesste Objekt des
            Referenzgazetteers hat 1'233 Repraesentantenzellen. Kleinere Werte
            begrenzen die Laufzeit grossflaechiger Objekte, aendern aber die
            Matrix: das Gewicht eines Objekts verteilt sich dann ueber weniger
            Zellen.
        k: Zahl der zugelassenen Distanzraenge (gekoppelt an max_slots).
        d_max: Suchradius in Ringschritten. Zwischen k und d_max liegt der
            Spielraum fuer leere Ringe.
        batch: Zahl der Quellzellen je Rechenblock (nur Laufzeit/Speicher).
        threads: DuckDB-Threads.
        index_memory_limit: DuckDB-Speicherlimit beim Bau des Stichprobenindex.
        weights_memory_limit: DuckDB-Speicherlimit bei der Ranggewichtung.
        named_only: Nur Objekte mit NAME und OBJEKTART beruecksichtigen.
    """

    r_star: int = 10
    cap: int = 99999
    k: int = 10
    d_max: int = 12
    batch: int = 40000
    threads: int = 4
    index_memory_limit: str = "5GB"
    weights_memory_limit: str = "2500MB"
    named_only: bool = True


def matrix_filename(measure: str) -> str:
    """Dateiname der Matrix-CSV eines Masses ('b1' -> 'b1_matrix.csv')."""
    if measure not in VALID_MEASURES:
        raise ValueError(
            f"Unbekanntes Assoziationsmass '{measure}' (erlaubt: {VALID_MEASURES})"
        )
    return f"{measure}_matrix.csv"


def b1_from_npmi(npmi: float, p_a: float, p_b: float) -> float:
    """B1(a->b) = NPMI(a,b) * p_b / (p_a + p_b)   (Formel 4.3).

    Der Faktor liegt zwischen 0 und 1 und waechst mit der Flaeche der
    Zielkategorie; er ist gesetzt und nicht aus den Daten abgeleitet.
    """
    denom = p_a + p_b
    if denom <= 0:
        return 0.0
    return npmi * (p_b / denom)


def npmi_from_W(W: np.ndarray) -> np.ndarray:
    """NPMI auf der Verbundverteilung einer gerichteten Gewichtsmatrix.

        p(A,B) = W[A,B] / sum(W);  p(A), p(B) = Zeilen-/Spaltenrandverteilung
        NPMI   = log2( p(A,B) / (p(A) p(B)) ) / ( -log2 p(A,B) )   in [-1, +1]

    W[A,B] ist gerichtet (Zeile = Quellkategorie, Spalte = Zielkategorie), also
    ist auch NPMI gerichtet. Zellen ohne Gewicht bekommen den Boden -1.

    Zeichengleich uebernommen aus
    experiment5_distance_association/2_analysis/common.py:137-153 — jene Fassung
    hat die Matrix npmi_dist_matrix_D1.csv gerechnet, auf die sich die in der
    Arbeit berichteten Zahlen stuetzen.
    """
    p = W / W.sum()
    pa = p.sum(1, keepdims=True)
    pb = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            p > 0,
            np.log2(p / (pa * pb)) / (-np.log2(np.where(p > 0, p, 1e-300))),
            -1.0,
        )
    return np.clip(np.nan_to_num(out, nan=-1.0), -1.0, 1.0)


def compute_measure(
    measure: str,
    db_path: str | Path,
    *,
    params=None,
    output_dir: Optional[str | Path] = None,
) -> "pd.DataFrame":
    """Dispatch auf die Berechnung des konfigurierten Masses.

    Args:
        measure: "b1" oder "d1".
        db_path: Pfad zur H3-DuckDB (Tabellen `features` und `h3_lookup`).
        params: Parameterobjekt des Masses — B1Params bzw. D1Params. Ohne
            Angabe gelten die Defaults des jeweiligen Masses.
        output_dir: Verzeichnis fuer die CSV (und bei D1 die Metadatei).

    Returns:
        Die Matrix als DataFrame (Index und Spalten = OBJEKTART).
    """
    if measure == B1:
        from .compute import compute_all

        p = params or B1Params()
        _, b1_df = compute_all(
            db_path,
            total_area_resolution=p.total_area_resolution,
            output_dir=output_dir,
        )
        return b1_df
    if measure == D1:
        from .d1 import compute_d1

        return compute_d1(db_path, params or D1Params(), output_dir=output_dir)
    raise ValueError(
        f"Unbekanntes Assoziationsmass '{measure}' (erlaubt: {VALID_MEASURES})"
    )
