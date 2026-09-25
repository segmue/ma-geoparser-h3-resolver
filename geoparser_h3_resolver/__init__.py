"""geoparser-h3-resolver — H3-basierte raeumliche Kontextsaetze fuer geoparser.

Die Namen werden lazy aufgeloest (PEP 562): `SpatialSentenceResolver` zieht
geoparser und torch nach, und die Assoziationsrechnung soll ohne beides
laufen — 'spatial-h3-assoc --measure d1' braucht nur DuckDB und h3.
`from geoparser_h3_resolver import SpatialSentenceResolver` funktioniert
unveraendert.
"""

__all__ = ["SpatialSentenceResolver", "BuildConfig", "StaticSlotConfig"]


def __getattr__(name):
    if name == "SpatialSentenceResolver":
        from .resolver import SpatialSentenceResolver

        return SpatialSentenceResolver
    if name == "BuildConfig":
        from .pipeline.build_config import BuildConfig

        return BuildConfig
    if name == "StaticSlotConfig":
        from .sentence_generator.config import StaticSlotConfig

        return StaticSlotConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
