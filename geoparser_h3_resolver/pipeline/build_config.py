"""
BuildConfig - Konfiguration fuer die geoparser-h3-resolver Build-Pipeline.

Liest eine config.yaml und stellt alle Parameter fuer:
  - H3-Konvertierung (target_cells, resolution, containment_mode)
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

    # Static Slots: welche OBJEKTARTs feste Slots im Satz bekommen
    static_slots: List[StaticSlotConfig] = field(default_factory=list)

    # Sentence Generator Parameter
    assoc_threshold: float = 0.001
    max_slots: int = 10
    max_slots_per_category: int = 5
    max_categories: int = 10
    max_filler_slots: int = 0

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

        return cls(
            gazetteer=raw.get("gazetteer", ""),
            source_crs=raw.get("source_crs", 2056),
            target_cells=raw.get("target_cells", 100),
            min_resolution=raw.get("min_resolution", 5),
            max_resolution=raw.get("max_resolution", 13),
            containment_mode=raw.get("containment_mode", "overlap"),
            output_file=raw.get("output_file", "spatial_h3.duckdb"),
            static_slots=static_slots,
            assoc_threshold=sg.get("assoc_threshold", 0.001),
            max_slots=sg.get("max_slots", 10),
            max_slots_per_category=sg.get("max_slots_per_category", 5),
            max_categories=sg.get("max_categories", 10),
            max_filler_slots=sg.get("max_filler_slots", 0),
        )

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def to_sentence_generator_config(self, matrix_path: Path) -> "SentenceGeneratorConfig":
        """Konvertiert zu SentenceGeneratorConfig fuer den Sentence Generator."""
        from ..sentence_generator import SentenceGeneratorConfig

        return SentenceGeneratorConfig(
            static_slots=list(self.static_slots),
            matrix_path=matrix_path,
            assoc_threshold=self.assoc_threshold,
            max_slots=self.max_slots,
            max_slots_per_category=self.max_slots_per_category,
            max_categories=self.max_categories,
            max_filler_slots=self.max_filler_slots,
        )

    def resolve_output_path(self, base_dir: Path) -> Path:
        """Loest den output_file Pfad auf."""
        p = Path(self.output_file)
        if p.is_absolute():
            return p
        return base_dir / p
