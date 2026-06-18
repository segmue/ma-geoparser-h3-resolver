"""
BatchSentenceGenerator - Batch-Vorberechnung von Beschreibungssaetzen.

Fuellt den Cache eines CandidateSentenceGenerator fuer viele Features auf
einmal, mit den set-basierten Batch-Queries der H3Engine statt per-Feature-
Queries. Die Einzel-Query-Semantik (Slot-Logik, Reihenfolgen) wird exakt
repliziert — generate() liefert danach fuer vorberechnete Features dieselben
Saetze als reine Cache-Hits.

Einschraenkung: Configs mit max_filler_slots > 0 werden nicht gebatcht
(precompute ist dann ein No-Op und der Einzelpfad greift); die Projekt-Configs
(config1/config2 und alle Ablations-Varianten) nutzen keine Filler-Slots.
"""

from typing import Dict, List, Tuple

from .generator import CandidateSentenceGenerator, FeatureInput, GeneratedSentence


class BatchSentenceGenerator:
    """Batch-Vorberechnung in den Cache eines CandidateSentenceGenerator."""

    def __init__(self, generator: CandidateSentenceGenerator):
        self.g = generator

    def precompute(self, features: List[FeatureInput]) -> None:
        """Berechnet Saetze fuer alle noch nicht gecachten Features vor."""
        g = self.g
        cfg = g.config
        if cfg.max_filler_slots > 0:
            return  # Filler-Pfad nicht gebatcht -> Einzelpfad nutzen

        seen: set[int] = set()
        todo: List[FeatureInput] = []
        for f in features:
            if f.feature_id in g._cache or f.feature_id in seen:
                continue
            seen.add(f.feature_id)
            todo.append(f)
        if not todo:
            return
        ids = [f.feature_id for f in todo]

        # Phase 1: Static Context — ein Batch-Query pro Static Slot.
        static_by_id: Dict[int, Dict[str, List[str]]] = {fid: {} for fid in ids}
        for slot in cfg.static_slots:
            try:
                df = g.engine.find_overlapping_features_batch(
                    ids, objektart=slot.objektart, max_results=slot.slots,
                ).df()
            except Exception:
                continue
            if df is None or df.empty:
                continue
            for src_id, grp in df.groupby("src_id", sort=False):
                names = grp["NAME"].dropna().tolist()
                if names:
                    static_by_id[int(src_id)][slot.label] = names

        # Phase 2: Dynamic Context — die assoziierten Ziel-Kategorien haengen
        # von der Quell-OBJEKTART ab, aber der teure Teil des Queries (h3_lookup-
        # Scans) nicht. Daher EIN Batch-Query ueber die Vereinigung aller
        # Kategorien-Listen; _allocate_context betrachtet pro Feature ohnehin
        # nur dessen eigene Kategorien, ueberzaehlige Zeilen sind unschaedlich.
        ctx_by_id: Dict[int, Dict[str, List[str]]] = {fid: {} for fid in ids}
        exclude_objektarts = {s.objektart for s in cfg.static_slots}
        assoc_by_objektart: Dict[str, List[Tuple[str, float]]] = {}
        for f in todo:
            if f.objektart not in assoc_by_objektart:
                associated = g._assoc_loader.get_associated_categories(
                    source_objektart=f.objektart,
                    threshold=cfg.assoc_threshold,
                    max_categories=cfg.max_categories,
                )
                assoc_by_objektart[f.objektart] = [
                    (cat, b1) for cat, b1 in associated
                    if cat not in exclude_objektarts
                ]

        dyn_feats = [f for f in todo if assoc_by_objektart[f.objektart]]
        union_categories = sorted({
            cat
            for f in dyn_feats
            for cat, _ in assoc_by_objektart[f.objektart]
        })
        if dyn_feats and union_categories:
            try:
                df = g.engine.find_intersecting_features_batch(
                    [f.feature_id for f in dyn_feats],
                    objektart_list=union_categories,
                    exclude_self=True,
                ).df()
            except Exception:
                df = None
            if df is not None and not df.empty:
                assoc_by_id = {f.feature_id: assoc_by_objektart[f.objektart]
                               for f in dyn_feats}
                for src_id, grp in df.groupby("src_id", sort=False):
                    ctx_by_id[int(src_id)] = self._allocate_context(
                        grp, assoc_by_id[int(src_id)],
                    )

        # Phase 3: Saetze bauen und cachen.
        for f in todo:
            static_context = static_by_id[f.feature_id]
            context_by_category = ctx_by_id[f.feature_id]
            sentence = g._template.build_sentence(
                name=f.name,
                objektart=f.objektart,
                context_by_category=context_by_category,
                filler_by_category={},
                static_context=static_context,
            )
            g._cache[f.feature_id] = GeneratedSentence(
                feature_id=f.feature_id,
                sentence=sentence,
                static_context=static_context,
                context_by_category=context_by_category,
                filler_by_category={},
                categories_used=list(context_by_category.keys()),
            )

    def _allocate_context(
        self,
        results_df,
        associated: List[Tuple[str, float]],
    ) -> Dict[str, List[str]]:
        """Slot-Vergabe pro Feature — exakt die Logik aus
        CandidateSentenceGenerator._find_dynamic_context (Schritt 3)."""
        cfg = self.g.config
        context_by_category: Dict[str, List[str]] = {}
        remaining_slots = cfg.max_slots
        max_per_cat = cfg.max_slots_per_category
        uuid_field = cfg.uuid_field

        for objektart, _ in associated:
            if remaining_slots <= 0:
                break
            slots = min(max_per_cat, remaining_slots)

            cat_df = results_df[results_df["OBJEKTART"] == objektart]
            if not cat_df.empty:
                unique_uuids = cat_df[uuid_field].unique()[:slots]
                selected = cat_df[cat_df[uuid_field].isin(unique_uuids)]
                names = selected["NAME"].tolist()
                if names:
                    context_by_category[objektart] = names
                    remaining_slots -= len(unique_uuids)

        return context_by_category
