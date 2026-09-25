"""
BuildConfig - Konfiguration fuer die geoparser-h3-resolver Build-Pipeline.

Liest eine config.yaml und stellt alle Parameter fuer:
  - H3-Konvertierung (target_cells, resolution, containment_mode)
  - Assoziation (Mass B1 oder D1, deren Rechenparameter, Matrixpfad)
  - Static Slots (OBJEKTART-basierte feste Slots im Satz)
  - Sentence Generator (assoc_threshold, max_slots, etc.)
  - Output (DuckDB-Pfad)

Die Geoparser-DB wird als Single Source of Truth behandelt: alle registrierten
Sources werden automatisch gelesen, keine Tabellen-/Spaltennamen in der Config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ..sentence_generator import SentenceGeneratorConfig

from ..association.measures import (
    B1,
    B1Params,
    D1,
    D1Params,
    VALID_MEASURES,
    matrix_filename,
)
from ..sentence_generator.config import StaticSlotConfig

# Pfad zum bundled configs-Verzeichnis
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class BuildConfig:
    """Vollstaendige Konfiguration fuer die Build-Pipeline.

    Wird aus einer config.yaml geladen oder per BuildConfig.for_gazetteer()
    fuer gebundelte Default-Configs.

    Example:
        # Gebundelte Default-Config:
        config = BuildConfig.for_gazetteer("swissnames3d")

        # Custom config:
        config = BuildConfig.from_yaml(Path("my_config.yaml"))
    """
    gazetteer: str
    source_crs: int = 2056
    target_cells: int = 100
    min_resolution: int = 5
    max_resolution: int = 13
    containment_mode: str = "overlap"
    output_file: str = "spatial_h3.duckdb"

    # Assoziation: welches Mass der Resolver liest und womit es gerechnet wird.
    # "b1" (Ueberlagerungsmass, Default) oder "d1" (Adjazenzmass).
    association_measure: str = B1
    matrix_path: Optional[str] = None
    b1: B1Params = field(default_factory=B1Params)
    d1: D1Params = field(default_factory=D1Params)

    # Static Slots: welche OBJEKTARTs feste Slots im Satz bekommen
    static_slots: List[StaticSlotConfig] = field(default_factory=list)

    # Sentence Generator Parameter
    assoc_threshold: float = 0.001
    max_slots: int = 10
    max_slots_per_category: int = 5
    max_categories: int = 10
    max_filler_slots: int = 0
    slot_allocation: str = "greedy"

    def __post_init__(self):
        if self.association_measure not in VALID_MEASURES:
            raise ValueError(
                f"Unbekanntes Assoziationsmass '{self.association_measure}' "
                f"(erlaubt: {VALID_MEASURES})"
            )

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        """Laedt BuildConfig aus einer config.yaml Datei."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config-Datei nicht gefunden: {path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls._from_dict(raw)

    @classmethod
    def for_gazetteer(cls, gazetteer_name: str) -> "BuildConfig":
        """Laedt die gebundelte Default-Config fuer einen Gazetteer."""
        config_path = _CONFIGS_DIR / f"{gazetteer_name}.yaml"
        if not config_path.exists():
            available = [p.stem for p in _CONFIGS_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Keine gebundelte Config fuer '{gazetteer_name}' gefunden. "
                f"Verfuegbar: {available}. "
                f"Eigene Config mit --config /pfad/config.yaml angeben."
            )
        return cls.from_yaml(config_path)

    @classmethod
    def _from_dict(cls, raw: dict) -> "BuildConfig":
        """Erstellt BuildConfig aus einem dict (geparster YAML-Inhalt)."""
        static_slots = [
            StaticSlotConfig(
                objektart=s["objektart"],
                label=s.get("label", s["objektart"]),
                slots=s.get("slots", 1),
            )
            for s in raw.get("static_slots", [])
        ]

        sg = raw.get("sentence_generator", {})
        assoc = raw.get("association", {}) or {}

        return cls(
            gazetteer=raw.get("gazetteer", ""),
            source_crs=raw.get("source_crs", 2056),
            target_cells=raw.get("target_cells", 100),
            min_resolution=raw.get("min_resolution", 5),
            max_resolution=raw.get("max_resolution", 13),
            containment_mode=raw.get("containment_mode", "overlap"),
            output_file=raw.get("output_file", "spatial_h3.duckdb"),
            association_measure=assoc.get("measure", B1),
            matrix_path=assoc.get("matrix_path"),
            b1=_params_from_dict(B1Params, assoc.get("b1", {}), "association.b1"),
            d1=_params_from_dict(D1Params, assoc.get("d1", {}), "association.d1"),
            static_slots=static_slots,
            assoc_threshold=sg.get("assoc_threshold", 0.001),
            max_slots=sg.get("max_slots", 10),
            max_slots_per_category=sg.get("max_slots_per_category", 5),
            max_categories=sg.get("max_categories", 10),
            max_filler_slots=sg.get("max_filler_slots", 0),
            slot_allocation=sg.get("slot_allocation", "greedy"),
        )

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def measure_params(self):
        """Parameterobjekt des konfigurierten Masses (B1Params oder D1Params)."""
        return self.d1 if self.association_measure == D1 else self.b1

    def matrix_filename(self) -> str:
        """Dateiname der Matrix des konfigurierten Masses."""
        return matrix_filename(self.association_measure)

    def resolve_matrix_path(self, base_dir: Path) -> Path:
        """Pfad zur Assoziationsmatrix.

        Ein explizit gesetzter `association.matrix_path` schlaegt das Mass;
        sonst liegt die Matrix als <measure>_matrix.csv neben der DuckDB.
        """
        if self.matrix_path:
            p = Path(self.matrix_path)
            return p if p.is_absolute() else Path(base_dir) / p
        return Path(base_dir) / self.matrix_filename()

    def to_sentence_generator_config(self, matrix_path: Path) -> "SentenceGeneratorConfig":
        """Konvertiert zu SentenceGeneratorConfig fuer den Sentence Generator."""
        from ..sentence_generator import SentenceGeneratorConfig

        return SentenceGeneratorConfig(
            static_slots=list(self.static_slots),
            matrix_path=matrix_path,
            measure=self.association_measure,
            assoc_threshold=self.assoc_threshold,
            max_slots=self.max_slots,
            max_slots_per_category=self.max_slots_per_category,
            max_categories=self.max_categories,
            max_filler_slots=self.max_filler_slots,
            slot_allocation=self.slot_allocation,
        )

    def resolve_output_path(self, base_dir: Path) -> Path:
        """Loest den output_file Pfad auf."""
        p = Path(self.output_file)
        if p.is_absolute():
            return p
        return base_dir / p


def _params_from_dict(cls, raw: dict, where: str):
    """Baut ein Parameter-Dataclass aus einem YAML-Teilbaum.

    Unbekannte Schluessel werden gemeldet statt stillschweigend verworfen — ein
    vertipptes `r_star` wuerde sonst unbemerkt den Default weiterlaufen lassen.
    """
    raw = raw or {}
    known = {f for f in cls.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"Unbekannte Schluessel in {where}: {unknown} "
            f"(erlaubt: {sorted(known)})"
        )
    return cls(**raw)
