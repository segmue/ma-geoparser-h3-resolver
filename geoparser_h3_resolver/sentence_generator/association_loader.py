"""
Laden und Cachen der B1 Association Matrix.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class AssociationMatrixLoader:
    """Laedt und cached die B1 Association Matrix.

    Die Matrix wird lazy geladen beim ersten Zugriff und dann im Speicher
    gehalten fuer schnelle wiederholte Abfragen.

    Attributes:
        matrix: pandas DataFrame mit OBJEKTART als Index und Spalten
    """

    def __init__(self, matrix_path: Path):
        """
        Initialisiert den Loader.

        Args:
            matrix_path: Pfad zur B1 Matrix CSV Datei
        """
        self._matrix_path = Path(matrix_path)
        self._matrix: Optional[pd.DataFrame] = None

    @property
    def matrix(self) -> pd.DataFrame:
        """Lazy-Load der B1 Matrix."""
        if self._matrix is None:
            self._matrix = self._load_matrix()
        return self._matrix

    def _load_matrix(self) -> pd.DataFrame:
        """Laedt die B1 Matrix aus CSV (Semicolon-separiert)."""
        if not self._matrix_path.exists():
            raise FileNotFoundError(f"B1 Matrix nicht gefunden: {self._matrix_path}")

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
        """Gibt assoziierte Kategorien zurueck, sortiert nach B1-Wert.

        Args:
            source_objektart: Die Quell-OBJEKTART
            threshold: Minimaler B1-Wert fuer Relevanz
            max_categories: Maximale Anzahl Kategorien

        Returns:
            Liste von (objektart, b1_wert) Tupeln, absteigend sortiert nach B1
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
