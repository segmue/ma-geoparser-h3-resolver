"""Tests fuer die Konfiguration des Assoziationsmasses (pipeline/build_config.py)."""

from pathlib import Path

import pytest
import yaml

from geoparser_h3_resolver.pipeline.build_config import BuildConfig

MINIMAL = {
    "gazetteer": "testgaz",
    "static_slots": [{"objektart": "Gemeindegebiet", "label": "Gemeinde", "slots": 2}],
}


def _write(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_ohne_association_block_bleibt_b1_das_mass(tmp_path):
    """Bestehende Configs ohne den neuen Block verhalten sich wie bisher."""
    config = BuildConfig.from_yaml(_write(tmp_path, MINIMAL))
    assert config.association_measure == "b1"
    assert config.matrix_path is None
    assert config.resolve_matrix_path(Path("/data")) == Path("/data/b1_matrix.csv")


def test_gebundelte_swissnames_config_ist_auf_b1(tmp_path):
    config = BuildConfig.for_gazetteer("swissnames3d")
    assert config.association_measure == "b1"
    assert config.d1.r_star == 10
    assert config.d1.k == 10
    assert config.d1.d_max == 12
    assert config.d1.cap == 99999
    assert config.b1.total_area_resolution == 10


def test_measure_d1_waehlt_die_d1_matrix(tmp_path):
    config = BuildConfig.from_yaml(_write(tmp_path, {**MINIMAL,
                                                    "association": {"measure": "d1"}}))
    assert config.resolve_matrix_path(Path("/data")) == Path("/data/d1_matrix.csv")
    assert config.matrix_filename() == "d1_matrix.csv"


def test_expliziter_matrix_path_schlaegt_das_mass(tmp_path):
    config = BuildConfig.from_yaml(_write(tmp_path, {
        **MINIMAL,
        "association": {"measure": "d1", "matrix_path": "/anderswo/meine.csv"},
    }))
    assert config.resolve_matrix_path(Path("/data")) == Path("/anderswo/meine.csv")


def test_relativer_matrix_path_liegt_neben_der_duckdb(tmp_path):
    config = BuildConfig.from_yaml(_write(tmp_path, {
        **MINIMAL,
        "association": {"matrix_path": "matrizen/b1_alt.csv"},
    }))
    assert config.resolve_matrix_path(Path("/data")) == Path("/data/matrizen/b1_alt.csv")


def test_d1_parameter_kommen_aus_der_yaml(tmp_path):
    config = BuildConfig.from_yaml(_write(tmp_path, {
        **MINIMAL,
        "association": {"measure": "d1",
                        "d1": {"r_star": 9, "k": 6, "d_max": 8, "named_only": False}},
    }))
    assert (config.d1.r_star, config.d1.k, config.d1.d_max) == (9, 6, 8)
    assert config.d1.named_only is False
    assert config.d1.cap == 99999       # nicht gesetzt -> Default
    assert config.measure_params() is config.d1


def test_unbekanntes_mass_wird_abgelehnt(tmp_path):
    with pytest.raises(ValueError, match="Assoziationsmass"):
        BuildConfig.from_yaml(_write(tmp_path, {**MINIMAL,
                                                "association": {"measure": "b2"}}))


def test_vertippter_d1_parameter_wird_gemeldet(tmp_path):
    """Ein stillschweigend verworfener Schluessel liesse den Default weiterlaufen."""
    with pytest.raises(ValueError, match="rstar"):
        BuildConfig.from_yaml(_write(tmp_path, {**MINIMAL,
                                                "association": {"d1": {"rstar": 9}}}))


def test_sentence_generator_config_traegt_das_mass(tmp_path):
    config = BuildConfig.from_yaml(_write(tmp_path, {**MINIMAL,
                                                    "association": {"measure": "d1"}}))
    sg = config.to_sentence_generator_config(Path("/data/d1_matrix.csv"))
    assert sg.measure == "d1"
    assert sg.assoc_threshold == 0.001
    assert sg.max_categories == 10
    assert sg.slot_allocation == "greedy"


def test_generator_achsen_kommen_aus_der_yaml(tmp_path):
    config = BuildConfig.from_yaml(_write(tmp_path, {
        **MINIMAL,
        "sentence_generator": {"assoc_threshold": 0.01, "max_categories": 20,
                               "slot_allocation": "proportional"},
    }))
    sg = config.to_sentence_generator_config(Path("/data/b1_matrix.csv"))
    assert sg.assoc_threshold == 0.01
    assert sg.max_categories == 20
    assert sg.slot_allocation == "proportional"
