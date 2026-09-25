"""
Adjazenzmass D1: Assoziation aus H3-Ringraengen statt aus Ueberlagerung.

Portiert aus experiment5_distance_association/2_analysis/ — dort haben
05_sample_index.py, 06b_neighbours_cellwise_det.py und 08_build_matrix.py am
17.09.2026 die Matrix npmi_dist_matrix_D1.csv gerechnet, auf die sich die in der
Arbeit berichteten D1-Zahlen stuetzen. Die Rechnung ist zeichengleich
uebernommen; neu sind nur die Parametrisierung, der Schrankentest als fester
Bestandteil und die Metadatei.

Das Verfahren in vier Schritten (Kapitel 4.2, Formeln 4.4 und 4.5):

  1. Stichprobenindex. Jede Zelle jedes Objekts wird auf die Ankeraufloesung r*
     abgebildet: feiner -> Elternzelle, gleich -> sie selbst, groeber ->
     Mittelpunkts-Kindzelle. Danach dedupliziert und je Objekt auf `cap` Zellen
     gedeckelt. Ein Punkt hat genau eine Zelle; fuer ihn ist die Konstruktion
     identisch zum einfachen Punktverfahren.
  2. Raenge. Je Quellzelle s werden die Ringe d = 0..D erzeugt und die Treffer
     nach minimaler Ringdistanz zu Raengen verdichtet. Es zaehlen hoechstens k
     Raenge; Rang 0 sind Objekte in derselben Zelle.
  3. Gewicht. w(a,s,b) = 1 / (m_r(a,s) * M_a): jeder gefuellte Rang wiegt gleich
     viel, und jedes Quellobjekt gibt insgesamt hoechstens k ab, unabhaengig von
     seiner Ausdehnung. Summiert ergibt das die gerichtete Gewichtsmatrix W
     (Zeile = Quellkategorie).
  4. NPMI auf W (measures.npmi_from_W, Formel 4.1).

Die Schranke aus Schritt 3 ist der schaerfste Selbsttest des Verfahrens: die
Zeilensumme einer Kategorie kann hoechstens das k-Fache ihrer Objektzahl sein.
Der Lauf vom 04.09.2026 verletzte sie in 40 von 110 Kategorien, weil die
Quellzellen nicht disjunkt aufgeteilt waren — deshalb prueft compute_d1() sie
jedes Mal und bricht bei Verletzung ab.

Gerechnet wird direkt auf den Tabellen `features` und `h3_lookup` der
H3-DuckDB; die H3Engine wird nicht gebraucht.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .measures import D1Params, npmi_from_W

__all__ = [
    "D1Params",
    "build_sample_index",
    "accumulate_weights",
    "check_weight_bound",
    "compute_d1",
]


def _configure(con, memory_limit: str, threads: int) -> None:
    """Speicherlimit, Threads und Auslagerungsverzeichnis setzen.

    Ohne gesetztes temp_directory legt DuckDB sein Spill-Verzeichnis `.tmp/` im
    aktuellen Arbeitsverzeichnis an — bei 34 Mio. Zellen sind das schnell
    Dutzende Megabyte im Projektordner.
    """
    spill = Path(tempfile.gettempdir()) / "duckdb_spill_geoparser_h3"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{memory_limit}';")
    con.execute(f"SET threads={threads};")
    con.execute(f"SET temp_directory='{spill}';")


# ─── H3-Bitarithmetik (vektorisiert, gegen h3-py geprueft) ───────────────────────
# Aufloesungsfeld Bit 52-55, 15 Ziffern zu 3 Bit ab Bit 44 abwaerts, unbenutzte
# Ziffern = 7. Uebernommen aus 05_sample_index.py:32-40.

def _res(h):
    return (h >> np.uint64(52)) & np.uint64(0xF)


def _setres(h, r):
    return (h & ~(np.uint64(0xF) << np.uint64(52))) | (np.uint64(r) << np.uint64(52))


def parent(h, r):
    """Elternzelle auf Aufloesung r (vektorisiert)."""
    low = np.uint64((1 << (3 * (15 - r))) - 1)
    return _setres(h, r) | low


def center_child(h, r, res_old):
    """Mittelpunkts-Kindzelle auf Aufloesung r (vektorisiert)."""
    m_old = (np.uint64(1) << (np.uint64(3) * (np.uint64(15) - res_old))) - np.uint64(1)
    m_new = np.uint64((1 << (3 * (15 - r))) - 1)
    return _setres(h, r) & ~(m_old ^ m_new)


def selftest(r: int = 10) -> None:
    """Prueft Eltern-/Kindberechnung gegen h3-py. Wirft AssertionError bei Abweichung."""
    import h3.api.basic_int as h3i

    for latlng, res in [((46.5, 8.0), 13), ((47.37, 8.54), 12),
                        ((46.0, 7.5), 6), ((46.8, 9.1), 8)]:
        c = h3i.latlng_to_cell(*latlng, res)
        a = np.array([c], np.uint64)
        if res > r:
            got, exp = int(parent(a, r)[0]), h3i.cell_to_parent(c, r)
        else:
            got, exp = int(center_child(a, r, _res(a))[0]), h3i.cell_to_center_child(c, r)
        assert got == exp, (res, got, exp)


def rings_for(cells, dmax: int):
    """(src_cell, cell, d) fuer d = 0..dmax, als drei numpy-Arrays."""
    import h3.api.numpy_int as h3n

    S, C, Dd = [], [], []
    for c in cells:
        S.append(np.array([c], np.uint64))
        C.append(np.array([c], np.uint64))
        Dd.append(np.zeros(1, np.uint8))
        for d in range(1, dmax + 1):
            r = h3n.grid_ring(int(c), d)
            if r.size == 0:
                continue
            r = r.astype(np.uint64)
            S.append(np.full(r.size, c, np.uint64))
            C.append(r)
            Dd.append(np.full(r.size, d, np.uint8))
    return np.concatenate(S), np.concatenate(C), np.concatenate(Dd)


# ─── Schritt 1: Stichprobenindex ────────────────────────────────────────────────

def _pool_sql(named_only: bool) -> str:
    where = ("WHERE NAME IS NOT NULL AND OBJEKTART IS NOT NULL"
             if named_only else "")
    return f"SELECT feature_id, OBJEKTART FROM features {where} ORDER BY feature_id"


def load_pool(db_path: str | Path, params: D1Params) -> pd.DataFrame:
    """Objektpool: feature_id und OBJEKTART, aufsteigend nach feature_id."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(_pool_sql(params.named_only)).df()
    finally:
        con.close()


def build_sample_index(db_path: str | Path, params: D1Params, verbose: bool = True):
    """Baut den Stichprobenindex (feature_id, cell) auf der Ankeraufloesung r*.

    Returns:
        pyarrow.Table mit den Spalten feature_id (int64) und cell (uint64).
    """
    import duckdb
    import pyarrow as pa

    selftest(params.r_star)
    r = params.r_star
    t0 = time.time()

    con = duckdb.connect(str(db_path), read_only=True)
    _configure(con, params.index_memory_limit, params.threads)
    where = ("WHERE NAME IS NOT NULL AND OBJEKTART IS NOT NULL"
             if params.named_only else "")
    con.execute(f"CREATE TEMP VIEW feat AS "
                f"SELECT feature_id, OBJEKTART, h3_resolution, h3_cell_count "
                f"FROM features {where}")
    if verbose:
        n_pool = con.execute("SELECT count(*) FROM feat").fetchone()[0]
        print(f"[pool] Objekte: {n_pool:,}")

    outs = []
    for lo in range(0, 16):
        t = con.execute(f"""SELECT l.feature_id, l.cell, l.cell_res FROM h3_lookup l
              JOIN feat f USING (feature_id) WHERE l.feature_id % 16 = {lo}""").fetch_arrow_table()
        fid = t.column("feature_id").to_numpy().astype(np.int64)
        cell = t.column("cell").to_numpy().astype(np.uint64)
        cres = t.column("cell_res").to_numpy().astype(np.uint64)
        del t
        rep = np.empty_like(cell)
        fine = cres > r
        coarse = cres < r
        same = ~(fine | coarse)
        rep[same] = cell[same]
        if fine.any():
            rep[fine] = parent(cell[fine], r)
        if coarse.any():
            rep[coarse] = center_child(cell[coarse], r, cres[coarse])
        outs.append((fid, rep))
        if verbose:
            print(f"  Block {lo+1}/16  {len(fid):,} Zellen  ({time.time()-t0:.0f}s)")
    con.close()

    fid = np.concatenate([o[0] for o in outs])
    rep = np.concatenate([o[1] for o in outs])
    del outs

    con = duckdb.connect(":memory:")
    _configure(con, params.index_memory_limit, params.threads)
    con.register("raw", pa.table({"feature_id": fid, "cell": rep}))
    idx = con.execute(f"""
        WITH d AS (SELECT DISTINCT feature_id, cell FROM raw),
        r AS (SELECT feature_id, cell,
                row_number() OVER (PARTITION BY feature_id ORDER BY hash(cell)) rn
              FROM d)
        SELECT feature_id, cell FROM r WHERE rn <= {params.cap}
    """).fetch_arrow_table()
    con.close()

    if verbose:
        print(f"[index] {idx.num_rows:,} Stichprobenzellen  ({time.time()-t0:.0f}s)")
    return idx


# ─── Schritt 2+3: Raenge und Gewichte ───────────────────────────────────────────

def accumulate_weights(sample_index, pool: pd.DataFrame, params: D1Params,
                       verbose: bool = True) -> tuple[np.ndarray, list[str], dict]:
    """Summiert die Ranggewichte zur gerichteten Kategorienmatrix W.

    Args:
        sample_index: pyarrow.Table (feature_id, cell) aus build_sample_index.
        pool: DataFrame (feature_id, OBJEKTART) aus load_pool.
        params: D1Params.

    Returns:
        (W, cats, meta) — W ist die gerichtete Gewichtsmatrix (Zeile =
        Quellkategorie), cats die Kategorien in Matrixordnung, meta die
        Laufkennzahlen (src_total, src_sha256, rank_fill_hist, W_total).
    """
    import duckdb
    import pyarrow as pa

    t0 = time.time()
    con = duckdb.connect(":memory:")
    _configure(con, params.weights_memory_limit, params.threads)
    con.register("sample_index", sample_index)
    con.register("pool", pa.Table.from_pandas(pool, preserve_index=False))
    con.execute("CREATE TABLE idx AS SELECT feature_id, cell FROM sample_index")
    con.execute("CREATE TABLE msrc AS SELECT feature_id, count(*) m FROM idx GROUP BY 1")
    con.execute("CREATE TABLE feat AS SELECT feature_id, OBJEKTART FROM pool")
    cats = [r[0] for r in con.execute(
        "SELECT DISTINCT OBJEKTART FROM feat ORDER BY 1").fetchall()]
    n = len(cats)
    con.execute("CREATE TABLE cat AS SELECT feature_id, OBJEKTART, "
                "dense_rank() OVER (ORDER BY OBJEKTART)-1 AS ci FROM feat")

    # Aufsteigend sortierte Quellzellenliste: macht jede Aufteilung nachweislich
    # disjunkt und den Lauf unabhaengig von der Ausfuehrungsordnung.
    src = con.execute("SELECT DISTINCT cell FROM idx ORDER BY cell") \
             .fetch_arrow_table().column("cell").to_numpy().astype(np.uint64)
    src = np.sort(src)
    src_total = int(src.size)
    src_sha = hashlib.sha256(src.tobytes()).hexdigest()
    if verbose:
        print(f"[liste] {src_total:,} Quellzellen · sha256 {src_sha}")
        print(f"[start] {n} Kategorien · k={params.k} D={params.d_max}", flush=True)

    W = np.zeros((n, n), np.float64)
    filled = np.zeros(params.k + 1, np.int64)

    for b0 in range(0, src_total, params.batch):
        chunk = src[b0:b0 + params.batch]
        S, C, D = rings_for(chunk, params.d_max)
        con.register("disk", pa.table({"s": S, "c": C, "d": D.astype(np.int16)}))
        con.execute("""CREATE OR REPLACE TEMP TABLE hits AS
            WITH h AS (SELECT k.feature_id AS sf, dk.s, i.feature_id AS tf, min(dk.d) AS d
                       FROM disk dk JOIN idx i ON i.cell = dk.c
                       JOIN idx k ON k.cell = dk.s
                       WHERE i.feature_id <> k.feature_id
                       GROUP BY 1,2,3),
            r AS (SELECT sf, s, tf, d, dense_rank() OVER (PARTITION BY sf, s ORDER BY d) rk FROM h)
            SELECT * FROM r WHERE rk <= ?""", [params.k])
        f = con.execute("SELECT nranks, count(*) FROM "
                        "(SELECT sf,s,max(rk) nranks FROM hits GROUP BY 1,2) "
                        "GROUP BY 1").fetchall()
        for nr, cnt in f:
            filled[min(int(nr), params.k)] += int(cnt)
        con.execute("""CREATE OR REPLACE TEMP TABLE v AS
            SELECT w.sf, w.tf, w.wq / m.m AS wgt FROM
              (SELECT h.sf, h.s, h.tf, 1.0/count(*) OVER (PARTITION BY h.sf,h.s,h.rk) AS wq FROM hits h) w
            JOIN msrc m ON m.feature_id = w.sf""")
        agg = con.execute("""
            SELECT cs.ci AS a, ct.ci AS b, sum(v.wgt) AS wsum
            FROM v JOIN cat cs ON cs.feature_id=v.sf JOIN cat ct ON ct.feature_id=v.tf
            GROUP BY 1,2""").fetch_arrow_table()
        np.add.at(W, (agg.column("a").to_numpy(), agg.column("b").to_numpy()),
                  agg.column("wsum").to_numpy())
        con.unregister("disk")
        if verbose:
            done = min(b0 + params.batch, src_total)
            print(f"  {done:,}/{src_total:,}  ({time.time()-t0:.0f}s)", flush=True)

    con.close()
    meta = {
        "src_total": src_total,
        "src_sha256": src_sha,
        "rank_fill_hist": filled.tolist(),
        "W_total": float(W.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    return W, cats, meta


# ─── Der Schrankentest ──────────────────────────────────────────────────────────

def check_weight_bound(W: np.ndarray, cats: list[str], object_counts: dict[str, int],
                       k: int, tol: float = 1e-9) -> dict:
    """Prueft: jedes Quellobjekt gibt hoechstens Gewicht k ab.

    Die Zeilensumme einer Kategorie kann damit hoechstens das k-Fache ihrer
    Objektzahl sein. Verletzungen heissen, dass Quellzellen doppelt gezaehlt
    wurden; dann darf aus W keine Matrix gebaut werden.

    Raises:
        ValueError: sobald eine Kategorie die Schranke ueberschreitet.
    """
    row_sums = W.sum(axis=1)
    limits = np.array([k * object_counts.get(c, 0) for c in cats], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(limits > 0, row_sums / limits, 0.0)

    verletzt = [(cats[i], float(row_sums[i]), float(limits[i]), float(ratio[i]))
                for i in range(len(cats)) if ratio[i] > 1.0 + tol]
    if verletzt:
        zeilen = "\n".join(
            f"  {c}: Zeilensumme {s:,.2f} bei Grenze {l:,.2f} (Verhaeltnis {r:.4f})"
            for c, s, l, r in sorted(verletzt, key=lambda x: -x[3])[:10])
        raise ValueError(
            f"Schrankentest verletzt in {len(verletzt)} von {len(cats)} Kategorien.\n"
            f"{zeilen}\n"
            "Ein Quellobjekt kann hoechstens Gewicht k abgeben; ein hoeherer Wert "
            "heisst, dass Quellzellen doppelt gezaehlt wurden. Aus diesem W darf "
            "keine Matrix gebaut werden.")

    return {
        "violations": 0,
        "max_ratio": float(ratio.max()) if ratio.size else 0.0,
        "median_ratio": float(np.median(ratio)) if ratio.size else 0.0,
    }


# ─── Schritt 4: die Matrix ──────────────────────────────────────────────────────

def compute_d1(
    db_path: str | Path,
    params: Optional[D1Params] = None,
    output_dir: Optional[str | Path] = None,
    sample_index_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Berechnet die D1-Matrix fuer alle OBJEKTART-Paare.

    Args:
        db_path: Pfad zur H3-DuckDB (Tabellen `features` und `h3_lookup`).
        params: D1Params; ohne Angabe die Defaults der Arbeit.
        output_dir: Optionaler Pfad fuer d1_matrix.csv und d1_matrix_meta.json.
        sample_index_path: Optionaler Pfad zu einem Stichprobenindex (Parquet).
            Existiert die Datei, wird sie gelesen statt neu gebaut; sonst wird
            der gebaute Index dorthin geschrieben. Nuetzlich, um einen Lauf
            gegen einen frueheren zu halten.
        verbose: Fortschritt auf die Konsole.

    Returns:
        Die D1-Matrix als DataFrame (Index und Spalten = OBJEKTART).
    """
    params = params or D1Params()
    t0 = time.time()

    pool = load_pool(db_path, params)
    if pool.empty:
        raise RuntimeError(f"Keine Objekte im Pool von {db_path}.")

    idx_path = Path(sample_index_path) if sample_index_path else None
    if idx_path is not None and idx_path.exists():
        import pyarrow.parquet as pq

        sample_index = pq.read_table(idx_path, columns=["feature_id", "cell"])
        if verbose:
            print(f"[index] gelesen aus {idx_path} ({sample_index.num_rows:,} Zeilen)")
    else:
        sample_index = build_sample_index(db_path, params, verbose=verbose)
        if idx_path is not None:
            import pyarrow.parquet as pq

            idx_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(sample_index, idx_path)
            if verbose:
                print(f"[index] geschrieben: {idx_path}")

    W, cats, meta = accumulate_weights(sample_index, pool, params, verbose=verbose)

    object_counts = pool.groupby("OBJEKTART").size().to_dict()
    bound = check_weight_bound(W, cats, object_counts, params.k)
    if verbose:
        print(f"[schranke] 0 Verletzungen · Hoechstverhaeltnis "
              f"{bound['max_ratio']:.4f} · Median {bound['median_ratio']:.4f}")

    M = npmi_from_W(W)
    df = pd.DataFrame(M, index=cats, columns=cats)

    off = ~np.eye(len(cats), dtype=bool)
    meta.update({
        "measure": "d1",
        "params": asdict(params),
        "db_path": str(db_path),
        "n_categories": len(cats),
        "n_objects": int(len(pool)),
        "bound_check": bound,
        "range": [float(M.min()), float(M.max())],
        "floor_share": float(np.mean(M[off] <= -0.999)),
        "diag_median": float(np.median(np.diag(M))),
        "total_seconds": round(time.time() - t0, 1),
    })

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "d1_matrix.csv"
        df.to_csv(csv_path, sep=";")
        meta["matrix_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        (out / "d1_matrix_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"\nMatrix gespeichert: {csv_path}")
            print(f"  sha256 {meta['matrix_sha256']}")

    if verbose:
        print(f"[fertig] {len(cats)}x{len(cats)} · W_total {meta['W_total']:,.2f} · "
              f"Bereich [{M.min():+.5f}, {M.max():+.5f}] · "
              f"Median-Diagonale {meta['diag_median']:+.5f} · "
              f"({meta['total_seconds']:.0f}s)")

    return df
