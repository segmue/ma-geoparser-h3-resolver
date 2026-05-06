"""
Konfiguration fuer den CandidateSentenceGenerator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class StaticSlotConfig:
    """Konfiguration fuer einen festen Slot im generierten Satz.

    Definiert welche OBJEKTART-Werte feste Slots bekommen (z.B. Gemeinde, Kanton).
    Die OBJEKTART-Werte stammen direkt aus der Geoparser-DB.

    Attributes:
        objektart: OBJEKTART-Wert aus der Geoparser-DB (z.B. 'Gemeindegebiet')
        label: Anzeige-Label im Satz (z.B. 'Gemeinde')
        slots: Maximale Anzahl Features im Satz
    """
    objektart: str
    label: str
    slots: int = 1


@dataclass
class SentenceGeneratorConfig:
    """Konfiguration fuer die Satzgenerierung.

    Attributes:
        assoc_threshold: Minimaler B1-Wert fuer relevante Kategorien
        max_slots: Maximale Anzahl Slots (distinct UUIDs) im Satz gesamt
        max_slots_per_category: Maximale Slots (distinct UUIDs) pro Kategorie
        max_categories: Maximale Anzahl Kategorien zu beruecksichtigen
        static_slots: Liste von festen Slots (nach OBJEKTART)
        matrix_path: Pfad zur B1 Matrix CSV (optional)
        category_separator: Trennzeichen zwischen Kategorien im Satz
        instance_separator: Trennzeichen zwischen Instanzen einer Kategorie
    """

    # Association thresholds
    assoc_threshold: float = 0.001

    # Slot allocation
    max_slots: int = 10
    max_slots_per_category: int = 5
    max_categories: int = 10
    max_filler_slots: int = 0

    # Static slots (ersetzt static_datasets)
    static_slots: List[StaticSlotConfig] = field(default_factory=list)
    uuid_field: str = "UUID"

    # Paths
    matrix_path: Optional[Path] = None

    # Template settings
    category_separator: str = "; "
    instance_separator: str = ", "

    def get_matrix_path(self) -> Path:
        """Gibt den Pfad zur B1 Matrix zurueck."""
        if self.matrix_path:
            return self.matrix_path
        raise FileNotFoundError(
            "matrix_path nicht gesetzt. Bitte SentenceGeneratorConfig mit "
            "matrix_path initialisieren oder spatial-h3-build ausfuehren."
        )

    @classmethod
    def default_swissnames(cls, matrix_path: Path) -> "SentenceGeneratorConfig":
        """Erstellt eine Standard-Config fuer swissnames3d."""
        return cls(
            static_slots=[
                StaticSlotConfig(objektart="Gemeindegebiet", label="Gemeinde", slots=2),
                StaticSlotConfig(objektart="Kanton", label="Kanton", slots=1),
                StaticSlotConfig(objektart="Bezirk", label="Bezirk", slots=1),
            ],
            matrix_path=matrix_path,
        )
