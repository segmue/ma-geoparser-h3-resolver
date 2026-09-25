"""Tests fuer das Adjazenzmass D1 (association/d1.py).

Die Rechnung laeuft auf einer winzigen, im Test gebauten DuckDB: drei Objekte
auf bekannten H3-Zellen, deren Ringabstaende von Hand nachvollziehbar sind.
"""

import duckdb
import h3.api.basic_int as h3i
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from geoparser_h3_resolver.association.d1 import (
    D1Params,
    _res,
    accumulate_weights,
    build_sample_index,
    center_child,
    check_weight_bound,
    compute_d1,
    parent,
    rings_for,
    selftest,
)

R_STAR = 10


# ── H3-Bitarithmetik ─────────────────────────────────────────────────────────

def test_selftest_laeuft_durch():
    selftest(R_STAR)


def test_parent_stimmt_mit_h3_py_ueberein():
    c = h3i.latlng_to_cell(46.5, 8.0, 13)
    got = int(parent(np.array([c], np.uint64), R_STAR)[0])
    assert got == h3i.cell_to_parent(c, R_STAR)


def test_center_child_stimmt_mit_h3_py_ueberein():
    c = h3i.latlng_to_cell(46.5, 8.0, 6)
    a = np.array([c], np.uint64)
    got = int(center_child(a, R_STAR, _res(a))[0])
    assert got == h3i.cell_to_center_child(c, R_STAR)


def test_rings_for_liefert_sechs_zellen_je_ringschritt():
    c = h3i.latlng_to_cell(46.5, 8.0, R_STAR)
    S, C, D = rings_for([np.uint64(c)], 2)
    assert (D == 0).sum() == 1        # die Quellzelle selbst
    assert (D == 1).sum() == 6
    assert (D == 2).sum() == 12
    assert set(S.tolist()) == {c}


# ── Der Schrankentest ────────────────────────────────────────────────────────

def test_schranke_haelt_bei_zulaessigem_gewicht():
    W = np.array([[0.0, 20.0], [5.0, 0.0]])
    bericht = check_weight_bound(W, ["X", "Y"], {"X": 2, "Y": 3}, k=10)
    assert bericht["violations"] == 0
    assert bericht["max_ratio"] == pytest.approx(1.0)


def test_schranke_schlaegt_bei_doppelzaehlung_an():
    """Zwei Objekte koennen bei k=10 hoechstens Gewicht 20 abgeben."""
    W = np.array([[0.0, 25.0], [5.0, 0.0]])
    with pytest.raises(ValueError, match="Schrankentest"):
        check_weight_bound(W, ["X", "Y"], {"X": 2, "Y": 3}, k=10)


# ── Eine kleine Gazetteer-DuckDB ─────────────────────────────────────────────

def _make_db(path, rows):
    """rows: [(feature_id, NAME, OBJEKTART, [cells], cell_res)]"""
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE features (
        feature_id INTEGER, UUID VARCHAR, NAME VARCHAR, OBJEKTART VARCHAR,
        source VARCHAR, h3_resolution TINYINT, h3_cell_count INTEGER)""")
    con.execute("CREATE TABLE h3_lookup (feature_id INTEGER, cell UBIGINT, cell_res TINYINT)")
    for fid, name, objektart, cells, res in rows:
        con.execute("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [fid, f"uuid-{fid}", name, objektart, "test", res, len(cells)])
        for cell in cells:
            con.execute("INSERT INTO h3_lookup VALUES (?, ?, ?)", [fid, cell, res])
    con.close()
    return path


@pytest.fixture
def drei_objekte(tmp_path):
    """A (Kategorie X) auf einer Zelle, B und C (Kategorie Y) auf Ring 1 und 2."""
    a = h3i.latlng_to_cell(46.5, 8.0, R_STAR)
    b = int(h3i.grid_ring(a, 1)[0])
    c = int(h3i.grid_ring(a, 2)[0])
    return _make_db(tmp_path / "t.duckdb", [
        (0, "A", "X", [a], R_STAR),
        (1, "B", "Y", [b], R_STAR),
        (2, "C", "Y", [c], R_STAR),
    ]), (a, b, c)


def test_gewicht_je_quellobjekt_und_rang_ist_eins(drei_objekte):
    """A sieht B auf Rang 1 und C auf Rang 2 — je Rang genau Gewicht 1."""
    db, _ = drei_objekte
    params = D1Params(r_star=R_STAR, k=3, d_max=3, batch=100)
    idx = build_sample_index(db, params, verbose=False)
    pool = pd.DataFrame({"feature_id": [0, 1, 2], "OBJEKTART": ["X", "Y", "Y"]})

    W, cats, meta = accumulate_weights(idx, pool, params, verbose=False)

    assert cats == ["X", "Y"]
    assert W[cats.index("X"), cats.index("Y")] == pytest.approx(2.0)
    assert meta["src_total"] == 3


def test_kein_quellobjekt_gibt_mehr_als_k_ab(drei_objekte):
    db, _ = drei_objekte
    params = D1Params(r_star=R_STAR, k=3, d_max=3, batch=100)
    idx = build_sample_index(db, params, verbose=False)
    pool = pd.DataFrame({"feature_id": [0, 1, 2], "OBJEKTART": ["X", "Y", "Y"]})

    W, cats, _ = accumulate_weights(idx, pool, params, verbose=False)

    check_weight_bound(W, cats, {"X": 1, "Y": 2}, k=params.k)


def test_flaechiges_objekt_verteilt_sein_gewicht_ueber_seine_zellen(tmp_path):
    """Ein Objekt mit M Zellen gibt je Zelle 1/M ab — nicht M-mal so viel."""
    a = h3i.latlng_to_cell(46.5, 8.0, R_STAR)
    flaeche = [a] + [int(x) for x in h3i.grid_ring(a, 1)]   # 7 Zellen
    ziel = int(h3i.grid_ring(a, 2)[0])
    db = _make_db(tmp_path / "t.duckdb", [
        (0, "Flaeche", "X", flaeche, R_STAR),
        (1, "Punkt", "Y", [ziel], R_STAR),
    ])
    params = D1Params(r_star=R_STAR, k=1, d_max=2, batch=100)
    idx = build_sample_index(db, params, verbose=False)
    pool = pd.DataFrame({"feature_id": [0, 1], "OBJEKTART": ["X", "Y"]})

    W, cats, _ = accumulate_weights(idx, pool, params, verbose=False)

    # Genau ein Rang zugelassen -> das Objekt gibt insgesamt hoechstens 1 ab.
    assert W[cats.index("X")].sum() <= 1.0 + 1e-12


def test_cap_deckelt_die_stichprobenzellen_je_objekt(tmp_path):
    a = h3i.latlng_to_cell(46.5, 8.0, R_STAR)
    viele = [a] + [int(x) for x in h3i.grid_disk(a, 3)]
    db = _make_db(tmp_path / "t.duckdb", [(0, "Gross", "X", viele, R_STAR)])

    idx = build_sample_index(db, D1Params(r_star=R_STAR, cap=5), verbose=False)

    assert idx.num_rows == 5


def test_feinere_zellen_werden_auf_die_ankeraufloesung_gehoben(tmp_path):
    """Zwei Res-13-Kinder derselben Res-10-Zelle ergeben eine Stichprobenzelle."""
    fein = h3i.latlng_to_cell(46.5, 8.0, 13)
    eltern = h3i.cell_to_parent(fein, R_STAR)
    kinder = [int(x) for x in h3i.cell_to_children(eltern, 13)][:2]
    db = _make_db(tmp_path / "t.duckdb", [(0, "Fein", "X", kinder, 13)])

    idx = build_sample_index(db, D1Params(r_star=R_STAR), verbose=False)

    assert idx.num_rows == 1
    assert int(idx.column("cell")[0].as_py()) == eltern


def test_objekte_ohne_namen_bleiben_draussen(tmp_path):
    a = h3i.latlng_to_cell(46.5, 8.0, R_STAR)
    b = int(h3i.grid_ring(a, 1)[0])
    db = _make_db(tmp_path / "t.duckdb", [
        (0, "A", "X", [a], R_STAR),
        (1, None, "Y", [b], R_STAR),
    ])

    idx = build_sample_index(db, D1Params(r_star=R_STAR), verbose=False)
    alle = build_sample_index(db, D1Params(r_star=R_STAR, named_only=False), verbose=False)

    assert idx.num_rows == 1
    assert alle.num_rows == 2


def test_compute_d1_schreibt_matrix_und_metadatei(drei_objekte, tmp_path):
    db, _ = drei_objekte
    out = tmp_path / "out"
    params = D1Params(r_star=R_STAR, k=3, d_max=3, batch=100)

    df = compute_d1(db, params, output_dir=out, verbose=False)

    assert list(df.index) == list(df.columns) == ["X", "Y"]
    assert df.to_numpy().min() >= -1.0 and df.to_numpy().max() <= 1.0
    assert (out / "d1_matrix.csv").exists()

    import json
    meta = json.loads((out / "d1_matrix_meta.json").read_text())
    assert meta["measure"] == "d1"
    assert meta["params"]["k"] == 3
    assert meta["bound_check"]["violations"] == 0
    assert len(meta["matrix_sha256"]) == 64


def test_compute_d1_liest_einen_vorhandenen_stichprobenindex(drei_objekte, tmp_path):
    db, _ = drei_objekte
    params = D1Params(r_star=R_STAR, k=3, d_max=3, batch=100)
    idx_path = tmp_path / "idx.parquet"

    compute_d1(db, params, sample_index_path=idx_path, verbose=False)
    assert idx_path.exists()

    # Zweiter Lauf liest die Datei statt sie neu zu bauen — gleiches Ergebnis.
    df2 = compute_d1(db, params, sample_index_path=idx_path, verbose=False)
    assert list(df2.index) == ["X", "Y"]
