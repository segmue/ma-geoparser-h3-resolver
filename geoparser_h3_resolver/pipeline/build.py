"""
Build-Pipeline: Geoparser SpatiaLite DB -> H3 DuckDB + B1 Matrix.

Liest ALLE registrierten Features aus der Geoparser-DB (auto-discovery via
source/gazetteer Metadaten), konvertiert zu H3 Cells, speichert in DuckDB
und berechnet Spatial Associations.

Die Geoparser-DB wird als Single Source of Truth behandelt — keine
Tabellennamen oder Spalten-Mappings in der Config noetig.

Verwendung:
    # Default swissnames3d (gebundelte Config):
    spatial-h3-build

    # Custom config:
    spatial-h3-build --config /pfad/zu/my_config.yaml

    # Als Python:
    from geoparser_h3_resolver.pipeline.build import build
    build()
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union

import appdirs
import duckdb
import pandas as pd
from shapely import wkb

from h3_spatial_engine import convert_geometry_to_h3, ContainmentMode

from .build_config import BuildConfig
from ..association import compute_all


def _get_geoparser_db_path() -> Path:
    """Findet Geoparsers Datenbank-Pfad via appdirs."""
    data_dir = Path(appdirs.user_data_dir("geoparser"))
    db_path = data_dir / "geoparser.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Geoparser-Datenbank nicht gefunden: {db_path}\n"
            "Bitte zuerst 'geoparser download <gazetteer>' ausfuehren."
        )
    return db_path


def _load_spatialite(conn: sqlite3.Connection) -> None:
    """Laedt die SpatiaLite-Extension."""
    conn.enable_load_extension(True)
    for lib in ["mod_spatialite", "libspatialite", "mod_spatialite.so", "mod_spatialite.dylib"]:
        try:
            conn.load_extension(lib)
            return
        except sqlite3.OperationalError:
            continue
    raise RuntimeError(
        "SpatiaLite Extension konnte nicht geladen werden. "
        "Bitte installieren: brew install spatialite-tools (macOS) "
        "oder apt install libspatialite-dev (Linux)"
    )


def _discover_sources(
    geoparser_db: Path,
    gazetteer_name: str,
) -> list[tuple[str, str]]:
    """Entdeckt alle registrierten Sources fuer einen Gazetteer.

    Liest aus Geoparsers internen Metadaten-Tabellen (source + gazetteer).

    Returns:
        Liste von (source_table_or_view, identifier_column) Tupeln
    """
    conn = sqlite3.connect(str(geoparser_db))

    rows = conn.execute("""
        SELECT s.name, s.location_id_name
        FROM source s
        JOIN gazetteer g ON s.gazetteer_id = g.id
        WHERE g.name = ?
    """, [gazetteer_name]).fetchall()

    conn.close()

    if not rows:
        raise RuntimeError(
            f"Keine Sources fuer Gazetteer '{gazetteer_name}' gefunden.\n"
            f"Ist der Gazetteer in der Geoparser-DB installiert?"
        )

    return rows


def _read_all_features(
    geoparser_db: Path,
    sources: list[tuple[str, str]],
) -> pd.DataFrame:
    """Liest ALLE Features aus allen Sources einer Geoparser-DB.

    Args:
        geoparser_db: Pfad zur Geoparser SQLite-DB
        sources: Liste von (source_name, id_column) aus _discover_sources()

    Returns:
        DataFrame mit Spalten: UUID, NAME, OBJEKTART, geometry (Shapely), source
    """
    conn = sqlite3.connect(str(geoparser_db))
    _load_spatialite(conn)

    records = []

    for source_name, id_col in sources:
        sql = f"""
            SELECT
                CAST("{id_col}" AS TEXT) as id_val,
                NAME,
                OBJEKTART,
                Hex(ST_AsBinary(geometry)) as geom_hex
            FROM "{source_name}"
            WHERE geometry IS NOT NULL
        """

        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.OperationalError as e:
            print(f"    Warnung: Source '{source_name}' nicht lesbar ({e}), uebersprungen.")
            continue

        for id_val, name_val, type_val, geom_hex in rows:
            if geom_hex is None:
                continue
            try:
                geom = wkb.loads(bytes.fromhex(geom_hex))
                records.append({
                    "UUID": id_val,
                    "NAME": name_val,
                    "OBJEKTART": type_val,
                    "geometry": geom,
                    "source": source_name,
                })
            except Exception:
                continue

        print(f"    {source_name}: {sum(1 for r in records if r['source'] == source_name)} Features")

    conn.close()

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["UUID", "NAME", "OBJEKTART", "geometry", "source"]
    )


def _convert_to_h3(
    df: pd.DataFrame,
    config: BuildConfig,
) -> pd.DataFrame:
    """Konvertiert Geometrien zu H3 Cells gemaess BuildConfig.

    Fuegt Spalten hinzu: h3_cells (list), h3_resolution (int), h3_cell_count (int)
    """
    containment_mode = ContainmentMode(config.containment_mode)
    h3_cells_list = []
    resolutions = []

    total = len(df)
    for i, row in df.iterrows():
        cells, res = convert_geometry_to_h3(
            row["geometry"],
            target_cells=config.target_cells,
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
            source_crs=config.source_crs,
            containment_mode=containment_mode,
        )
        h3_cells_list.append(list(cells))
        resolutions.append(res)

        if (i + 1) % 5000 == 0:
            print(f"  Konvertiert: {i+1}/{total}")

    result = df.copy()
    result["h3_cells"] = h3_cells_list
    result["h3_resolution"] = resolutions
    result["h3_cell_count"] = [len(c) for c in h3_cells_list]
    return result


def _create_duckdb(output_path: Path, all_features: pd.DataFrame) -> None:
    """Erstellt die DuckDB mit features und h3_lookup Tabellen."""
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL h3 FROM community; LOAD h3;")

    all_features = all_features.reset_index(drop=True)
    all_features["feature_id"] = range(len(all_features))

    def cells_to_uint64(cells):
        return [int(c, 16) if isinstance(c, str) else int(c) for c in cells]

    all_features["h3_cells_uint"] = all_features["h3_cells"].apply(cells_to_uint64)

    conn.execute("""
        CREATE TABLE features (
            feature_id INTEGER PRIMARY KEY,
            UUID VARCHAR,
            NAME VARCHAR,
            OBJEKTART VARCHAR,
            source VARCHAR,
            h3_cells UBIGINT[],
            h3_resolution TINYINT,
            h3_cell_count INTEGER
        )
    """)

    for _, row in all_features.iterrows():
        conn.execute("""
            INSERT INTO features (feature_id, UUID, NAME, OBJEKTART, source,
                                  h3_cells, h3_resolution, h3_cell_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["feature_id"],
            row.get("UUID"),
            row.get("NAME"),
            row.get("OBJEKTART"),
            row.get("source"),
            row["h3_cells_uint"],
            row["h3_resolution"],
            row["h3_cell_count"],
        ])

    print("  Erstelle h3_lookup Index...")
    conn.execute("""
        CREATE TABLE h3_lookup AS
        SELECT DISTINCT
            feature_id,
            UNNEST(h3_cells) as cell,
            h3_resolution as cell_res
        FROM features
        ORDER BY cell
    """)

    conn.execute("CREATE INDEX idx_lookup_cell ON h3_lookup(cell)")
    conn.execute("CREATE INDEX idx_lookup_feature ON h3_lookup(feature_id)")
    conn.execute("CREATE INDEX idx_features_uuid ON features(UUID)")

    total_features = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    total_cells = conn.execute("SELECT COUNT(*) FROM h3_lookup").fetchone()[0]
    print(f"  DuckDB erstellt: {total_features} Features, {total_cells} Lookup-Eintraege")

    conn.close()


def build(
    config: Union[BuildConfig, str, Path, None] = None,
    gazetteer_db_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Vollstaendige Build-Pipeline.

    Args:
        config: BuildConfig, Pfad zu einer config.yaml, oder None fuer swissnames3d Default
        gazetteer_db_path: Pfad zu Geoparsers DB. Default: appdirs
        output_path: Pfad fuer Output-DuckDB. Ueberschreibt output_file aus config.

    Returns:
        Path zur erstellten DuckDB
    """
    # Config aufloesen
    if config is None:
        build_config = BuildConfig.for_gazetteer("swissnames3d")
    elif isinstance(config, (str, Path)):
        build_config = BuildConfig.from_yaml(config)
    else:
        build_config = config

    # Pfade aufloesen
    if gazetteer_db_path is None:
        gazetteer_db_path = _get_geoparser_db_path()
    else:
        gazetteer_db_path = Path(gazetteer_db_path)

    base_dir = Path(appdirs.user_data_dir("geoparser"))
    base_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = build_config.resolve_output_path(base_dir)
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_path.parent

    print("=" * 60)
    print(f"geoparser-h3-resolver Build Pipeline ({build_config.gazetteer})")
    print("=" * 60)
    print(f"  Geoparser DB: {gazetteer_db_path}")
    print(f"  Output DuckDB: {output_path}")
    print(f"  H3: target_cells={build_config.target_cells}, "
          f"res={build_config.min_resolution}-{build_config.max_resolution}, "
          f"mode={build_config.containment_mode}")
    print()

    # Step 1: Sources auto-discovern + Features lesen
    print("Step 1: Entdecke Sources und lese Features...")
    sources = _discover_sources(gazetteer_db_path, build_config.gazetteer)
    print(f"  {len(sources)} Sources gefunden:")
    for name, id_col in sources:
        print(f"    - {name} (id={id_col})")

    print()
    all_features = _read_all_features(gazetteer_db_path, sources)
    print(f"\n  Total: {len(all_features)} Features")

    if all_features.empty:
        raise RuntimeError(
            "Keine Features gefunden. Ist die Geoparser DB korrekt befuellt?\n"
            f"DB-Pfad: {gazetteer_db_path}"
        )

    # Step 2: H3-Konvertierung
    print(f"\nStep 2: Konvertiere Geometrien zu H3 Cells...")
    all_features = _convert_to_h3(all_features, build_config)
    total_cells = all_features["h3_cell_count"].sum()
    print(f"  Total H3 Cells: {total_cells:,}")

    # Step 3: DuckDB erstellen
    print(f"\nStep 3: Erstelle DuckDB...")
    _create_duckdb(output_path, all_features)

    # Step 4: Spatial Associations berechnen
    print(f"\nStep 4: Berechne Spatial Associations...")
    compute_all(db_path=str(output_path), output_dir=str(matrix_dir))

    print("\n" + "=" * 60)
    print("Build abgeschlossen!")
    print(f"  DuckDB:     {output_path}")
    print(f"  B1 Matrix:  {matrix_dir / 'b1_matrix.csv'}")
    print("=" * 60)

    return output_path


def main():
    """CLI Entrypoint mit --config und --gazetteer Optionen."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build H3 DuckDB from Geoparser gazetteer data"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Pfad zu einer config.yaml (default: bundled swissnames3d config)",
    )
    parser.add_argument(
        "--gazetteer", "-g",
        type=str,
        default=None,
        help="Gazetteer-Name fuer gebundelte Config (z.B. swissnames3d)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Pfad zur Geoparser-DB (default: appdirs)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Pfad fuer Output-DuckDB (ueberschreibt output_file aus config)",
    )
    args = parser.parse_args()

    # Config bestimmen
    if args.config:
        config = args.config
    elif args.gazetteer:
        config = BuildConfig.for_gazetteer(args.gazetteer)
    else:
        config = None  # default: swissnames3d

    build(config=config, gazetteer_db_path=args.db, output_path=args.output)


if __name__ == "__main__":
    main()
