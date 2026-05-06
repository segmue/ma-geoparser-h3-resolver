"""
CandidateSentenceGenerator - Generiert beschreibende Saetze fuer Gazetteer-Features.

Zwei-Phasen-Generierung:
  Phase 1: Static Context (z.B. Gemeinde, Kanton) — fixe Slots nach OBJEKTART
  Phase 2: Dynamic Context (via B1 Association Matrix) — proportionale Slot-Vergabe

Beispiel-Output:
    Alpiner Gipfel "Matterhorn" in Zermatt (Gemeinde), Wallis (Kanton). Bei Zmuttgrat (Grat); Theodulstrasse (Strasse)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .config import SentenceGeneratorConfig
from .association_loader import AssociationMatrixLoader
from .templates import SentenceTemplate

if TYPE_CHECKING:
    from h3_spatial_engine import H3Engine


@dataclass
class FeatureInput:
    """Input-Feature fuer die Satzgenerierung.

    Attributes:
        feature_id: Eindeutige ID des Features in der DuckDB
        name: Name des Features (kann None sein)
        objektart: OBJEKTART/Kategorie des Features
    """
    feature_id: int
    name: Optional[str]
    objektart: str


@dataclass
class GeneratedSentence:
    """Ergebnis der Satzgenerierung.

    Attributes:
        feature_id: ID des Quell-Features
        sentence: Der generierte Beschreibungssatz
        static_context: Dict von {Label: [Namen]} der statischen Kontext-Instanzen
        context_by_category: Dict von {OBJEKTART: [Namen]} der assoziierten Kontext-Instanzen
        filler_by_category: Dict von {OBJEKTART: [Namen]} der Filler-Instanzen (restliche Slots)
        categories_used: Liste der verwendeten dynamischen Kategorien
    """
    feature_id: int
    sentence: str
    static_context: Dict[str, List[str]]
    context_by_category: Dict[str, List[str]]
    filler_by_category: Dict[str, List[str]]
    categories_used: List[str]


class CandidateSentenceGenerator:
    """Generiert beschreibende Saetze fuer Gazetteer-Features.

    Algorithmus:
    1. Static Context: Pro konfiguriertem Dataset die ueberlappenden Features finden
    2. Dynamic Context: Relevante Kategorien aus B1-Matrix, Slots proportional verteilen
    3. EINE Query fuer alle dynamischen Kategorien via h3_lookup Index
    4. Ergebnisse nach Slots aufteilen
    5. Satz aus Template bauen

    Example:
        from h3_spatial_engine import H3Engine
        from geoparser_h3_resolver.sentence_generator import CandidateSentenceGenerator, FeatureInput

        engine = H3Engine("data/spatial_h3.duckdb")
        generator = CandidateSentenceGenerator(engine)

        feature = FeatureInput(feature_id=123, name="Matterhorn", objektart="Alpiner Gipfel")
        result = generator.generate(feature)
        print(result.sentence)
    """

    def __init__(
        self,
        engine: "H3Engine",
        config: Optional[SentenceGeneratorConfig] = None
    ):
        self.engine = engine
        self.config = config or SentenceGeneratorConfig()

        matrix_path = self.config.get_matrix_path()
        self._assoc_loader = AssociationMatrixLoader(matrix_path)
        self._template = SentenceTemplate(self.config)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(self, feature: FeatureInput) -> GeneratedSentence:
        """Generiert einen beschreibenden Satz fuer ein einzelnes Feature.

        Args:
            feature: Das Quell-Feature

        Returns:
            GeneratedSentence mit Satz und Kontext-Informationen
        """
        # Phase 1: Static Context
        static_context = self._find_static_context(feature.feature_id)

        # Phase 2: Dynamic Context (association-based + filler)
        context_by_category, filler_by_category = self._find_dynamic_context(feature)

        # Phase 3: Satz bauen
        sentence = self._template.build_sentence(
            name=feature.name,
            objektart=feature.objektart,
            context_by_category=context_by_category,
            filler_by_category=filler_by_category,
            static_context=static_context,
        )

        return GeneratedSentence(
            feature_id=feature.feature_id,
            sentence=sentence,
            static_context=static_context,
            context_by_category=context_by_category,
            filler_by_category=filler_by_category,
            categories_used=list(context_by_category.keys()),
        )

    def generate_batch(
        self,
        features: List[FeatureInput]
    ) -> List[GeneratedSentence]:
        """Generiert Saetze fuer mehrere Features."""
        return [self.generate(feature) for feature in features]

    # -------------------------------------------------------------------------
    # Static Context
    # -------------------------------------------------------------------------

    def _find_static_context(
        self,
        feature_id: int,
    ) -> Dict[str, List[str]]:
        """Findet statischen Kontext nach OBJEKTART (z.B. Gemeinde, Kanton).

        Returns:
            Dict von {Label: [Namen]}, z.B. {'Gemeinde': ['Zermatt'], 'Kanton': ['Wallis']}
        """
        static_context: Dict[str, List[str]] = {}

        for slot in self.config.static_slots:
            try:
                results_df = self.engine.find_overlapping_features(
                    feature_id=feature_id,
                    objektart=slot.objektart,
                    max_results=slot.slots,
                ).df()
            except Exception:
                continue

            if results_df is not None and not results_df.empty:
                names = results_df["NAME"].dropna().tolist()
                if names:
                    static_context[slot.label] = names

        return static_context

    # -------------------------------------------------------------------------
    # Dynamic Context (Association-based)
    # -------------------------------------------------------------------------

    def _find_dynamic_context(
        self,
        feature: FeatureInput,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Findet dynamischen Kontext via B1 Association Matrix + Filler.

        Returns:
            Tuple von (association_context, filler_context)
            Beide sind Dict von {OBJEKTART: [Namen]}
        """
        # 1. Assoziierte Kategorien aus B1-Matrix holen
        associated = self._assoc_loader.get_associated_categories(
            source_objektart=feature.objektart,
            threshold=self.config.assoc_threshold,
            max_categories=self.config.max_categories,
        )

        # Static-Slot-OBJEKTARTs ausschliessen (werden separat behandelt)
        exclude_objektarts = {s.objektart for s in self.config.static_slots}
        associated = [(cat, b1) for cat, b1 in associated
                      if cat not in exclude_objektarts]

        context_by_category: Dict[str, List[str]] = {}
        used_feature_ids: list[int] = []

        if associated:
            # 2. ALLE assoziierten Kategorien abfragen
            objektart_list = [cat for cat, _ in associated]
            try:
                results_df = self.engine.find_intersecting_features(
                    feature_id=feature.feature_id,
                    objektart_list=objektart_list,
                    exclude_id=feature.feature_id,
                ).df()
            except Exception:
                results_df = None

            # 3. Ergebnisse befuellen + ungenutzte Slots umverteilen
            if results_df is not None and not results_df.empty:
                remaining_slots = self.config.max_slots
                max_per_cat = self.config.max_slots_per_category
                uuid_field = self.config.uuid_field

                for objektart, _ in associated:
                    if remaining_slots <= 0:
                        break
                    slots = min(max_per_cat, remaining_slots)

                    mask = results_df["OBJEKTART"] == objektart
                    cat_df = results_df[mask]
                    if not cat_df.empty:
                        unique_uuids = cat_df[uuid_field].unique()[:slots]
                        selected = cat_df[cat_df[uuid_field].isin(unique_uuids)]
                        names = selected["NAME"].tolist()
                        if names:
                            context_by_category[objektart] = names
                            used_feature_ids.extend(selected["feature_id"].tolist())
                            remaining_slots -= len(unique_uuids)

        # 4. Verbleibende Slots mit kleinsten intersecting Features auffuellen
        filler_by_category: Dict[str, List[str]] = {}
        used_slots = sum(len(names) for names in context_by_category.values())
        remaining_slots = min(
            self.config.max_filler_slots,
            self.config.max_slots - used_slots,
        )

        if remaining_slots > 0:
            try:
                exclude_ids = [feature.feature_id] + used_feature_ids
                filler_df = self.engine.find_intersecting_features(
                    feature_id=feature.feature_id,
                    exclude_ids=exclude_ids,
                    order_by_size=True,
                    max_results=remaining_slots,
                ).df()
            except Exception:
                filler_df = None

            if filler_df is not None and not filler_df.empty:
                for _, row in filler_df.iterrows():
                    objektart = row["OBJEKTART"]
                    name = row["NAME"]
                    if objektart not in filler_by_category:
                        filler_by_category[objektart] = []
                    filler_by_category[objektart].append(name)

        return context_by_category, filler_by_category

    # -------------------------------------------------------------------------
    # Slot Allocation
    # -------------------------------------------------------------------------

    def _allocate_slots(
        self,
        associated: List[Tuple[str, float]]
    ) -> Dict[str, int]:
        """Verteilt Instanz-Slots proportional nach Assoziationsstaerke."""
        if not associated:
            return {}

        total_slots = self.config.max_slots
        max_per_cat = self.config.max_slots_per_category

        total_weight = sum(b1 for _, b1 in associated)
        if total_weight <= 0:
            return {}

        allocation = {}
        for cat, b1 in associated:
            raw = (b1 / total_weight) * total_slots
            allocation[cat] = min(max_per_cat, max(0, int(round(raw))))

        return allocation

    def get_available_categories(self) -> List[str]:
        """Gibt alle verfuegbaren OBJEKTART-Kategorien zurueck."""
        return self._assoc_loader.get_all_categories()
