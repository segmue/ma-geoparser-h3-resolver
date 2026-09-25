"""
Laden und Cachen der Assoziationsmatrix.

Welches Mass in der Datei steht — B1 (Ueberlagerungsmass) oder D1
(Adjazenzmass) — ist hier gleichgueltig: beide haben dasselbe Format
(quadratisch, `;`-separiert, OBJEKTART als Index und Kopfzeile, Zeile =
Quellkategorie) und werden gleich gelesen. Die Auswahl trifft die Config ueber
den Pfad.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class AssociationMatrixLoader:
    """Laedt und cached die Assoziationsmatrix.

    Die Matrix wird lazy geladen beim ersten Zugriff und dann im Speicher
    gehalten fuer schnelle wiederholte Abfragen.

    Attributes:
        matrix: pandas DataFrame mit OBJEKTART als Index und Spalten
    """

    def __init__(self, matrix_path: Path):
        """
        Initialisiert den Loader.

        Args:
            matrix_path: Pfad zur Matrix-CSV (b1_matrix.csv oder d1_matrix.csv)
        """
        self._matrix_path = Path(matrix_path)
        self._matrix: Optional[pd.DataFrame] = None

    @property
    def matrix(self) -> pd.DataFrame:
        """Lazy-Load der Matrix."""
        if self._matrix is None:
            self._matrix = self._load_matrix()
        return self._matrix

    def _load_matrix(self) -> pd.DataFrame:
        """Laedt die Matrix aus CSV (Semicolon-separiert)."""
        if not self._matrix_path.exists():
            raise FileNotFoundError(
                f"Assoziationsmatrix nicht gefunden: {self._matrix_path}\n"
                "b1_matrix.csv entsteht bei 'spatial-h3-build', "
                "d1_matrix.csv bei 'spatial-h3-assoc --measure d1'."
            )

        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    self._matrix_path,
                    sep=";",
                    index_col=0,
                    encoding=encoding
                )
                return df
            except UnicodeDecodeError:
                continue

        df = pd.read_csv(
            self._matrix_path,
            sep=";",
            index_col=0,
            encoding='utf-8',
            errors='replace'
        )
        return df

    def get_associated_categories(
        self,
        source_objektart: str,
        threshold: float,
        max_categories: int
    ) -> List[Tuple[str, float]]:
        """Gibt assoziierte Kategorien zurueck, sortiert nach Assoziationswert.

        Der Schwellwert wirkt vor der Sortierung, die Kappung danach: beide
        aendern die Laenge der Liste, nie die Rangfolge.

        Args:
            source_objektart: Die Quell-OBJEKTART
            threshold: Minimaler Assoziationswert fuer Relevanz
            max_categories: Maximale Anzahl Kategorien

        Returns:
            Liste von (objektart, wert) Tupeln, absteigend sortiert
        """
        if source_objektart not in self.matrix.index:
            return []

        row = self.matrix.loc[source_objektart]

        candidates = []
        for col in row.index:
            if col == source_objektart:
                continue
            try:
                val = float(row[col])
                if val >= threshold:
                    candidates.append((col, val))
            except (ValueError, TypeError):
                continue

        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:max_categories]

    def get_all_categories(self) -> List[str]:
        """Gibt alle verfuegbaren OBJEKTART-Kategorien zurueck."""
        return list(self.matrix.index)

    def reload(self) -> None:
        """Laedt die Matrix neu (z.B. nach Aenderungen)."""
        self._matrix = None
        _ = self.matrix
