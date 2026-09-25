"""Tests fuer die Slot-Vergabe (sentence_generator/allocation.py)."""

import pytest

from geoparser_h3_resolver.sentence_generator.allocation import (
    allocate,
    allocate_greedy,
    allocate_proportional,
)

# Projekt-Konfiguration (config1/config2)
MAX_SLOTS = 10
MAX_PER_CAT = 5


def test_greedy_erschoepft_budget_mit_den_obersten_kategorien():
    """Der dokumentierte Effekt: zwei Kategorien verbrauchen alles."""
    assoc = [("A", 0.9), ("B", 0.5), ("C", 0.4), ("D", 0.1)]
    avail = {"A": 20, "B": 20, "C": 20, "D": 20}
    assert allocate_greedy(assoc, avail, MAX_SLOTS, MAX_PER_CAT) == {"A": 5, "B": 5}


def test_greedy_ueberspringt_leere_kategorien_ohne_budget_zu_verbrauchen():
    assoc = [("A", 0.9), ("B", 0.5), ("C", 0.4)]
    avail = {"A": 2, "B": 0, "C": 20}
    assert allocate_greedy(assoc, avail, MAX_SLOTS, MAX_PER_CAT) == {"A": 2, "C": 5}


def test_proportional_streut_breiter_als_greedy():
    assoc = [("A", 0.4), ("B", 0.3), ("C", 0.2), ("D", 0.1)]
    avail = {c: 20 for c, _ in assoc}
    alloc = allocate_proportional(assoc, avail, MAX_SLOTS, MAX_PER_CAT)
    assert alloc == {"A": 4, "B": 3, "C": 2, "D": 1}
    assert sum(alloc.values()) == MAX_SLOTS


def test_proportional_haelt_obergrenze_je_kategorie():
    assoc = [("A", 0.95), ("B", 0.05)]
    avail = {"A": 20, "B": 20}
    alloc = allocate_proportional(assoc, avail, MAX_SLOTS, MAX_PER_CAT)
    assert alloc["A"] == MAX_PER_CAT
    assert sum(alloc.values()) <= MAX_SLOTS


def test_proportional_verteilt_nicht_fuellbare_slots_um():
    """A bekaeme rechnerisch 9, kann aber nur 2 — der Rest geht an B und C."""
    assoc = [("A", 0.9), ("B", 0.05), ("C", 0.05)]
    avail = {"A": 2, "B": 20, "C": 20}
    alloc = allocate_proportional(assoc, avail, MAX_SLOTS, MAX_PER_CAT)
    assert alloc["A"] == 2
    assert sum(alloc.values()) == MAX_SLOTS
    assert alloc["B"] > 0 and alloc["C"] > 0


def test_beide_modi_ueberschreiten_das_budget_nie():
    assoc = [(chr(65 + i), 1.0 / (i + 1)) for i in range(10)]
    avail = {c: 20 for c, _ in assoc}
    for fn in (allocate_greedy, allocate_proportional):
        alloc = fn(assoc, avail, MAX_SLOTS, MAX_PER_CAT)
        assert sum(alloc.values()) <= MAX_SLOTS
        assert all(0 < n <= MAX_PER_CAT for n in alloc.values())


def test_budget_wird_durch_verfuegbarkeit_begrenzt():
    assoc = [("A", 0.6), ("B", 0.4)]
    avail = {"A": 1, "B": 1}
    for fn in (allocate_greedy, allocate_proportional):
        assert fn(assoc, avail, MAX_SLOTS, MAX_PER_CAT) == {"A": 1, "B": 1}


def test_nullgewichte_bekommen_keine_slots():
    assoc = [("A", 0.0), ("B", 0.5)]
    avail = {"A": 20, "B": 20}
    assert "A" not in allocate_proportional(assoc, avail, MAX_SLOTS, MAX_PER_CAT)


def test_leere_eingaben():
    assert allocate([], {}, MAX_SLOTS, MAX_PER_CAT) == {}
    assert allocate([("A", 1.0)], {"A": 5}, 0, MAX_PER_CAT) == {}
    assert allocate([("A", 1.0)], {"A": 5}, MAX_SLOTS, 0) == {}


def test_dispatch_default_ist_greedy():
    assoc = [("A", 0.5), ("B", 0.5), ("C", 0.5)]
    avail = {c: 20 for c, _ in assoc}
    assert allocate(assoc, avail, MAX_SLOTS, MAX_PER_CAT) == \
        allocate_greedy(assoc, avail, MAX_SLOTS, MAX_PER_CAT)


def test_dispatch_unbekannter_modus():
    with pytest.raises(ValueError):
        allocate([("A", 1.0)], {"A": 5}, MAX_SLOTS, MAX_PER_CAT, mode="zufall")
