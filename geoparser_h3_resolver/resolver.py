"""
SpatialSentenceResolver - Geoparser Resolver mit H3-basierter raeumlicher Kontextgenerierung.

Erbt von SentenceTransformerResolver und ueberschreibt _generate_description()
um statt einfacher Admin-Hierarchie-Beschreibungen raeumlich-informierte Saetze
basierend auf H3 Spatial Associations zu generieren.

Verwendung:
    from geoparser import Geoparser, SpacyRecognizer
    from geoparser_h3_resolver import SpatialSentenceResolver

    # Default (bundled swissnames3d config):
    resolver = SpatialSentenceResolver(gazetteer_name="swissnames3d")

    # Custom config:
    resolver = SpatialSentenceResolver(config_path="/path/to/my_config.yaml")

    gp = Geoparser(recognizer=SpacyRecognizer(), resolver=resolver)
    docs = gp.parse("Das Matterhorn liegt in den Walliser Alpen.")
"""

from pathlib import Path
from typing import Optional, Union


from geoparser.modules.resolvers.sentencetransformer import SentenceTransformerResolver
from h3_multi_resolution_engine import H3Engine

from .sentence_generator import (
    CandidateSentenceGenerator,
    FeatureInput,
    SentenceGeneratorConfig,
)
from .pipeline.build_config import BuildConfig


class SpatialSentenceResolver(SentenceTransformerResolver):
    """Resolver mit H3-basierter raeumlicher Kontextgenerierung.

    Generiert Beschreibungen wie:
        Alpiner Gipfel "Matterhorn" bei Zmuttgrat, Hoernligrat (Grat);
        Theodulstrasse (Strasse). In Zermatt (Gemeinde), Wallis (Kanton)

    Statt der einfachen Beschreibung:
        Matterhorn (Alpiner Gipfel) in Zermatt, Wallis, Wallis
    """

    def __init__(
        self,
        model_name: str = "dguzh/geo-all-MiniLM-L6-v2",
        gazetteer_name: str = "swissnames3d",
        min_similarity: float = 0.6,
        max_tiers: int = 3,
        attribute_map: dict = None,
        # Spatial-spezifisch:
        config_path: Optional[Union[str, Path]] = None,
        duckdb_path: Optional[Union[str, Path]] = None,
        sentence_config: Optional[SentenceGeneratorConfig] = None,
    ):
        """
        Args:
            model_name: HuggingFace Transformer Modell
            gazetteer_name: Name des Gazetteers (default: swissnames3d)
            min_similarity: Minimale Cosine-Similarity fuer Matches
            max_tiers: Maximale Suchstufen
            attribute_map: Custom Attribute-Map (optional)
            config_path: Pfad zu einer config.yaml. Wenn angegeben, werden
                         duckdb_path und sentence_config daraus abgeleitet
                         (sofern nicht explizit uebergeben).
            duckdb_path: Pfad zur H3 DuckDB. Ueberschreibt config_path-Ableitung.
            sentence_config: Konfiguration fuer den Sentence Generator.
                             Ueberschreibt config_path-Ableitung.
        """
        super().__init__(
            model_name=model_name,
            gazetteer_name=gazetteer_name,
            min_similarity=min_similarity,
            max_tiers=max_tiers,
            attribute_map=attribute_map,
        )

        from geoparser.db.db import DATABASE_URL as _GEOPARSER_DATABASE_URL
        base_dir = Path(_GEOPARSER_DATABASE_URL.replace("sqlite:///", "")).parent

        # Lade BuildConfig wenn config_path angegeben, sonst bundled Default
        if config_path is not None:
            build_config = BuildConfig.from_yaml(config_path)
        else:
            try:
                build_config = BuildConfig.for_gazetteer(gazetteer_name)
            except FileNotFoundError:
                build_config = None

        # DuckDB-Pfad bestimmen
        if duckdb_path is not None:
            resolved_duckdb = Path(duckdb_path)
        elif build_config is not None:
            resolved_duckdb = build_config.resolve_output_path(base_dir)
        else:
            resolved_duckdb = self._default_duckdb_path()

        self._engine = H3Engine(resolved_duckdb)

        # SentenceGeneratorConfig bestimmen
        if sentence_config is not None:
            resolved_sentence_config = sentence_config
        elif build_config is not None:
            matrix_path = resolved_duckdb.parent / "b1_matrix.csv"
            resolved_sentence_config = build_config.to_sentence_generator_config(matrix_path)
        else:
            matrix_path = resolved_duckdb.parent / "b1_matrix.csv"
            resolved_sentence_config = SentenceGeneratorConfig.default_swissnames(matrix_path)

        self._generator = CandidateSentenceGenerator(self._engine, resolved_sentence_config)
        self._uuid_cache: dict[str, tuple[int, str, str] | None] = {}

    def _generate_description(self, candidate) -> str:
        """Override: Generiert raeumlich-informierte Beschreibung.

        Verwendet den H3-basierten CandidateSentenceGenerator fuer Candidates
        die in der DuckDB gefunden werden. Faellt zurueck auf die einfache
        Admin-Hierarchie-Beschreibung bei unbekannten Candidates.
        """
        uuid = candidate.location_id_value

        # UUID -> (feature_id, NAME, OBJEKTART) Lookup (cached)
        if uuid not in self._uuid_cache:
            row = self._engine.conn.execute(
                "SELECT feature_id, NAME, OBJEKTART FROM features WHERE UUID = ? LIMIT 1",
                [uuid],
            ).fetchone()
            self._uuid_cache[uuid] = (row[0], row[1], row[2]) if row else None

        cached = self._uuid_cache[uuid]
        if cached is None:
            return super()._generate_description(candidate)

        feature_id, name, objektart = cached
        feature_input = FeatureInput(
            feature_id=feature_id,
            name=name,
            objektart=objektart,
        )

        try:
            return self._generator.generate(feature_input).sentence
        except Exception:
            return super()._generate_description(candidate)

    @staticmethod
    def _default_duckdb_path() -> Path:
        """Standard-Pfad: im selben Verzeichnis wie geoparser's Daten."""
        from geoparser.db.db import DATABASE_URL as _GEOPARSER_DATABASE_URL
        return Path(_GEOPARSER_DATABASE_URL.replace("sqlite:///", "")).parent / "spatial_h3.duckdb"
