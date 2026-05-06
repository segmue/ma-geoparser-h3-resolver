"""
Satz-Templates und Formatierung.

Format mit Static Context:
    {OBJEKTART} "{NAME}" in {Static1} ({Label1}), {Static2} ({Label2}). Bei {Inst1} ({Kat1}); ...

Beispiel:
    Alpiner Gipfel "Matterhorn" in Zermatt (Gemeinde), Wallis (Kanton). Bei Zmuttgrat (Grat); Theodulstrasse (Strasse)
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SentenceGeneratorConfig


class SentenceTemplate:
    """Formatiert Features und Kontext zu beschreibenden Saetzen."""

    def __init__(self, config: "SentenceGeneratorConfig"):
        self.config = config

    def format_feature(self, name: str, objektart: str) -> str:
        """Formatiert das Haupt-Feature.

        Returns:
            Formatierter String wie 'Alpiner Gipfel "Matterhorn"'
        """
        if name:
            return f'{objektart} "{name}"'
        return objektart

    def format_category_group(
        self,
        objektart: str,
        instance_names: List[str]
    ) -> str:
        """Formatiert eine Gruppe von Instanzen einer Kategorie.

        Returns:
            Formatierter String wie 'Zmuttgrat, Hoernligrat (Grat)'
        """
        if not instance_names:
            return ""

        names_str = self.config.instance_separator.join(instance_names)
        return f"{names_str} ({objektart})"

    def build_sentence(
        self,
        name: str,
        objektart: str,
        context_by_category: Optional[Dict[str, List[str]]] = None,
        filler_by_category: Optional[Dict[str, List[str]]] = None,
        static_context: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Baut den kompletten beschreibenden Satz.

        Reihenfolge:
            1. Assoziations-basierter Kontext (context_by_category)
            2. Filler-Kontext (filler_by_category, restliche Slots)
            3. Statischer Kontext (static_context, z.B. Gemeinde/Kanton)

        Returns:
            Kompletter Satz
        """
        feature_part = self.format_feature(name, objektart)

        parts = []

        # 1. Assoziations-basierter Kontext
        if context_by_category:
            items = []
            for cat, names in context_by_category.items():
                if names:
                    formatted = self.format_category_group(cat, names)
                    if formatted:
                        items.append(formatted)
            if items:
                parts.append("bei " + self.config.category_separator.join(items))

        # 2. Filler-Kontext (restliche Slots, kleinste Features)
        if filler_by_category:
            items = []
            for cat, names in filler_by_category.items():
                if names:
                    formatted = self.format_category_group(cat, names)
                    if formatted:
                        items.append(formatted)
            if items:
                parts.append("nahe " + self.config.category_separator.join(items))

        # 3. Statischer Kontext (Gemeinde, Kanton, etc.)
        if static_context:
            static_items = []
            for label, names in static_context.items():
                for n in names:
                    static_items.append(f"{n} ({label})")
            if static_items:
                parts.append("in " + self.config.instance_separator.join(static_items))

        if not parts:
            return feature_part

        return feature_part + " " + ". ".join(parts)
