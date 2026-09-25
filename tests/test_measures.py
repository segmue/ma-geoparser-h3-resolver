"""Tests fuer die Assoziationsmasse (association/measures.py)."""

import numpy as np
import pytest

from geoparser_h3_resolver.association.compute import calculate_npmi
from geoparser_h3_resolver.association.measures import (
    B1,
    D1,
    VALID_MEASURES,
    b1_from_npmi,
    compute_measure,
    matrix_filename,
    npmi_from_W,
)


# ── NPMI (Formel 4.1) ────────────────────────────────────────────────────────

def test_npmi_ohne_gemeinsame_flaeche_ist_der_boden():
    assert calculate_npmi(0.5, 0.5, 0.0) == -1.0


def test_npmi_bei_unabhaengigkeit_ist_null():
    """p_ab = p_a * p_b ist genau der Erwartungswert bei Unabhaengigkeit."""
    assert calculate_npmi(0.5, 0.5, 0.25) == 0.0


def test_npmi_bei_vollstaendiger_deckung_ist_eins():
    assert calculate_npmi(0.5, 0.5, 0.5) == 1.0


def test_npmi_ist_symmetrisch():
    assert calculate_npmi(0.2, 0.6, 0.15) == calculate_npmi(0.6, 0.2, 0.15)


# ── B1 (Formel 4.3) ──────────────────────────────────────────────────────────

def test_b1_daempft_mit_der_flaeche_der_zielkategorie():
    """Der Faktor p_b/(p_a+p_b) liegt zwischen 0 und 1 und waechst mit p_b."""
    assert b1_from_npmi(1.0, 0.1, 0.3) == pytest.approx(0.75)
    assert b1_from_npmi(1.0, 0.3, 0.1) == pytest.approx(0.25)


def test_b1_ist_auf_der_diagonale_der_halbe_npmi():
    assert b1_from_npmi(0.8, 0.25, 0.25) == pytest.approx(0.4)


def test_b1_ohne_flaeche_ist_null():
    assert b1_from_npmi(-1.0, 0.0, 0.0) == 0.0


# ── NPMI auf der Gewichtsmatrix (D1) ─────────────────────────────────────────

def test_npmi_from_W_trennt_perfekt_getrennte_kategorien():
    """Gewicht nur auf der Diagonale: Diagonale +1, alles daneben am Boden."""
    W = np.array([[2.0, 0.0], [0.0, 2.0]])
    M = npmi_from_W(W)
    assert M[0, 0] == pytest.approx(1.0)
    assert M[1, 1] == pytest.approx(1.0)
    assert M[0, 1] == -1.0
    assert M[1, 0] == -1.0


def test_npmi_from_W_ist_null_bei_gleichverteiltem_gewicht():
    M = npmi_from_W(np.ones((2, 2)))
    assert np.allclose(M, 0.0)


def test_npmi_from_W_bleibt_im_wertebereich():
    rng = np.random.default_rng(7)
    M = npmi_from_W(rng.random((6, 6)) * 100)
    assert M.min() >= -1.0 and M.max() <= 1.0


def test_npmi_from_W_ist_gerichtet():
    """Zeile = Quellkategorie: eine unsymmetrische W bleibt unsymmetrisch."""
    W = np.array([[5.0, 4.0], [1.0, 5.0]])
    M = npmi_from_W(W)
    assert M[0, 1] != pytest.approx(M[1, 0])


# ── Registry ─────────────────────────────────────────────────────────────────

def test_matrix_filename_je_mass():
    assert matrix_filename(B1) == "b1_matrix.csv"
    assert matrix_filename(D1) == "d1_matrix.csv"


def test_matrix_filename_lehnt_unbekanntes_mass_ab():
    with pytest.raises(ValueError):
        matrix_filename("b2")


def test_compute_measure_lehnt_unbekanntes_mass_ab():
    with pytest.raises(ValueError):
        compute_measure("b2", "egal.duckdb")


def test_valid_measures_sind_b1_und_d1():
    assert VALID_MEASURES == ("b1", "d1")
