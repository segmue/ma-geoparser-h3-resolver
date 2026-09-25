"""
Slot-Vergabe fuer den dynamischen Kontext des CandidateSentenceGenerator.

Zwei Strategien, ueber SentenceGeneratorConfig.slot_allocation waehlbar:

  "greedy" (Default, bisheriges Verhalten)
      Kategorien in der Reihenfolge absteigender Assoziationsstaerke bedienen,
      jede bis zu max_slots_per_category, bis das Budget max_slots erschoepft
      ist. Folge: bei max_slots=10 und max_slots_per_category=5 verbrauchen die
      obersten zwei belegbaren Kategorien das ganze Budget.

  "proportional"
      Slots proportional zum Assoziationsgewicht der zugelassenen Kategorien,
      gedeckelt durch max_slots_per_category und die tatsaechliche Verfuegbarkeit.
      Nicht platzierbare Slots werden iterativ auf Kategorien mit Restkapazitaet
      umverteilt (groesster-Rest-Rundung). Kein Auffuellen ueber das Budget hinaus.

Beide Funktionen sind rein: sie bekommen die verfuegbare Instanzzahl je Kategorie
und geben die Slotzahl je Kategorie zurueck. Die Reihenfolge der Kategorien und
die Auswahl der Instanzen innerhalb einer Kategorie sind nicht ihre Aufgabe und
bleiben unveraendert — dadurch ist "greedy" bitgleich zum bisherigen Pfad.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Tuple

GREEDY = "greedy"
PROPORTIONAL = "proportional"
VALID_MODES = (GREEDY, PROPORTIONAL)


def allocate_greedy(
    associated: List[Tuple[str, float]],
    available: Mapping[str, int],
    max_slots: int,
    max_per_cat: int,
) -> Dict[str, int]:
    """Gierige Vergabe in der Reihenfolge von `associated` (bisheriges Verhalten)."""
    alloc: Dict[str, int] = {}
    remaining = max_slots
    for cat, _ in associated:
        if remaining <= 0:
            break
        n = min(max_per_cat, remaining, available.get(cat, 0))
        if n > 0:
            alloc[cat] = n
            remaining -= n
    return alloc


def allocate_proportional(
    associated: List[Tuple[str, float]],
    available: Mapping[str, int],
    max_slots: int,
    max_per_cat: int,
) -> Dict[str, int]:
    """Proportional zum Gewicht, mit iterativer Umverteilung nicht fuellbarer Slots."""
    cap = {cat: min(max_per_cat, available.get(cat, 0)) for cat, _ in associated}
    alloc = {cat: 0 for cat, _ in associated}
    remaining = min(max_slots, sum(cap.values()))
    active = [(cat, w) for cat, w in associated if cap[cat] > 0 and w > 0]

    while remaining > 0 and active:
        total_w = sum(w for _, w in active)
        if total_w <= 0:
            break
        shares = [(cat, (w / total_w) * remaining, cap[cat] - alloc[cat])
                  for cat, w in active]

        placed = 0
        for cat, ideal, room in shares:
            n = min(room, int(math.floor(ideal)))
            if n > 0:
                alloc[cat] += n
                placed += n

        left = remaining - placed
        if left > 0:
            # Restslots nach groesstem Bruchteil, nur wo noch Platz ist.
            frac = sorted(
                ((cat, ideal - math.floor(ideal)) for cat, ideal, _ in shares
                 if cap[cat] - alloc[cat] > 0),
                key=lambda x: x[1],
                reverse=True,
            )
            for cat, _ in frac:
                if left <= 0:
                    break
                alloc[cat] += 1
                left -= 1
                placed += 1

        remaining -= placed
        active = [(cat, w) for cat, w in active if cap[cat] - alloc[cat] > 0 and w > 0]
        if placed == 0:
            break

    return {cat: n for cat, n in alloc.items() if n > 0}


def allocate(
    associated: List[Tuple[str, float]],
    available: Mapping[str, int],
    max_slots: int,
    max_per_cat: int,
    mode: str = GREEDY,
) -> Dict[str, int]:
    """Dispatch auf die konfigurierte Strategie.

    Args:
        associated: [(OBJEKTART, Assoziationsgewicht)], absteigend sortiert.
        available: Anzahl tatsaechlich verfuegbarer Instanzen (distinct UUIDs)
            je OBJEKTART. Fehlende Eintraege gelten als 0.
        max_slots: Gesamtbudget an Instanz-Slots.
        max_per_cat: Obergrenze je Kategorie.
        mode: "greedy" oder "proportional".

    Returns:
        {OBJEKTART: Slotzahl} — nur Kategorien mit mindestens einem Slot.
    """
    if not associated or max_slots <= 0 or max_per_cat <= 0:
        return {}
    if mode == PROPORTIONAL:
        return allocate_proportional(associated, available, max_slots, max_per_cat)
    if mode == GREEDY:
        return allocate_greedy(associated, available, max_slots, max_per_cat)
    raise ValueError(f"Unbekannte slot_allocation '{mode}' (erlaubt: {VALID_MODES})")
