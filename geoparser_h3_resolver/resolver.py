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
        engine=None,
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
            engine: Optionale Engine-Injektion (duck-typed wie H3Engine, z.B.
                    GeometricEngine aus Experiment 3). Erfordert ein explizites
                    sentence_config; config_path/duckdb_path werden dann ignoriert.
        """
        super().__init__(
            model_name=model_name,
            gazetteer_name=gazetteer_name,
            min_similarity=min_similarity,
            max_tiers=max_tiers,
            attribute_map=attribute_map,
        )

        if engine is not None:
            if sentence_config is None:
                raise ValueError(
                    "engine=-Injektion erfordert ein explizites sentence_config."
                )
            self._engine = engine
            self._generator = CandidateSentenceGenerator(self._engine, sentence_config)
            self._uuid_cache = {}
            self._search_cache = {}
            return

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
        # (text, method, tiers) -> Kandidatenliste; Gazetteer-Suche ist
        # deterministisch, wiederholte Toponyme kosten so nur einen Dict-Lookup.
        self._search_cache: dict[tuple[str, str, int], list] = {}

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
                "SELECT feature_id, NAME, OBJEKTART FROM features WHERE UUID = ? "
                "ORDER BY feature_id LIMIT 1",
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

    def precompute_descriptions(self, candidates) -> None:
        """Batch-Vorberechnung fuer viele Kandidaten (UUID-Lookup + Saetze).

        Fuellt _uuid_cache (ein IN-Query statt N Einzelqueries) und den
        Generator-Cache (set-basierte Batch-Queries), sodass nachfolgende
        _generate_description()-Aufrufe reine Cache-Hits sind. Optional —
        ohne Aufruf bleibt der Einzelpfad voll funktionsfaehig.
        """
        from .sentence_generator.batch_generator import BatchSentenceGenerator

        todo_uuids = []
        seen_uuids = set()
        for c in candidates:
            u = c.location_id_value
            if u not in self._uuid_cache and u not in seen_uuids:
                seen_uuids.add(u)
                todo_uuids.append(u)

        for i in range(0, len(todo_uuids), 1000):
            chunk = todo_uuids[i:i + 1000]
            placeholders = ", ".join("?" for _ in chunk)
            rows = self._engine.conn.execute(
                f"""
                SELECT UUID, feature_id, NAME, OBJEKTART FROM features
                WHERE UUID IN ({placeholders})
                QUALIFY ROW_NUMBER() OVER (PARTITION BY UUID ORDER BY feature_id) = 1
                """,
                chunk,
            ).fetchall()
            found = {r[0]: (r[1], r[2], r[3]) for r in rows}
            for u in chunk:
                self._uuid_cache[u] = found.get(u)

        feats = []
        seen_ids = set()
        for c in candidates:
            cached = self._uuid_cache.get(c.location_id_value)
            if cached is None:
                continue
            feature_id, name, objektart = cached
            if feature_id in seen_ids or feature_id in self._generator._cache:
                continue
            seen_ids.add(feature_id)
            feats.append(FeatureInput(
                feature_id=feature_id, name=name, objektart=objektart,
            ))
        if feats:
            BatchSentenceGenerator(self._generator).precompute(feats)

    def _gather_candidates(self, texts, references, candidates, results, method, tiers):
        """Override: wie Parent, aber gazetteer.search pro (Text, Methode, Tier)
        memoized — wiederholte Toponym-Texte (haeufig in Korpora) suchen nur einmal.
        """
        for text, doc_references, doc_candidates, doc_results in zip(
            texts, references, candidates, results
        ):
            for ref_idx, ((start, end), result) in enumerate(
                zip(doc_references, doc_results)
            ):
                if result is not None:
                    continue

                reference_text = text[start:end]
                key = (reference_text, method, tiers)
                if key not in self._search_cache:
                    self._search_cache[key] = self.gazetteer.search(
                        reference_text, method, tiers=tiers
                    )
                existing_ids = {c.id for c in doc_candidates[ref_idx]}
                for candidate in self._search_cache[key]:
                    if candidate.id not in existing_ids:
                        doc_candidates[ref_idx].append(candidate)

    def _embed_candidates(self, candidates, results):
        """Override: fuellt vor dem Parent-Lauf den Batch-Pfad.

        Der Parent generiert Beschreibungen einzeln pro Kandidat (Einzelpfad,
        ~5 DuckDB-Queries/Feature). Hier werden die eindeutigen, noch nicht
        embeddeten Kandidaten vorab in einem Rutsch durch
        precompute_descriptions() gejagt; der unveraenderte Parent-Code findet
        danach nur noch Cache-Hits vor.
        """
        todo = {}
        for doc_candidates, doc_results in zip(candidates, results):
            for candidate_list, result in zip(doc_candidates, doc_results):
                if result is not None:
                    continue
                for c in candidate_list:
                    if c.id not in self.candidate_embeddings:
                        todo[c.id] = c
        if todo:
            self.precompute_descriptions(list(todo.values()))
        super()._embed_candidates(candidates, results)

    @staticmethod
    def _default_duckdb_path() -> Path:
        """Standard-Pfad: im selben Verzeichnis wie geoparser's Daten."""
        from geoparser.db.db import DATABASE_URL as _GEOPARSER_DATABASE_URL
        return Path(_GEOPARSER_DATABASE_URL.replace("sqlite:///", "")).parent / "spatial_h3.duckdb"
