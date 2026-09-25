"""
spatial-h3-assoc — Assoziationsmatrizen rechnen und gegeneinander halten.

Der Befehl trennt die Assoziationsrechnung vom Build: `spatial-h3-build` legt
die DuckDB an und rechnet dabei B1; D1 (Adjazenzmass) laeuft hier, auf einer
bestehenden DuckDB, ohne Encoder und ohne Geoparser-Modelle.

    # D1 mit den Vorgabewerten der Arbeit (r*=10, cap=99999, k=10, D=12)
    spatial-h3-assoc --measure d1

    # D1 mit abweichenden Parametern
    spatial-h3-assoc --measure d1 --r-star 9 --k 6 --out-dir ./matrizen

    # B1 neu rechnen (was auch Step 4 des Builds tut)
    spatial-h3-assoc --measure b1 --overwrite

    # Eichung: wie viele Kontextkategorien nennt welches Mass bei welcher
    # Schwelle und welcher Kappung?
    spatial-h3-assoc --report

Rangfolge der Werte: CLI schlaegt YAML, YAML schlaegt Code-Default.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..association.measures import B1, D1, VALID_MEASURES, matrix_filename
from .build_config import BuildConfig


def _geoparser_base_dir() -> Path:
    """Verzeichnis, in dem geoparser seine Daten haelt (dort liegt die DuckDB)."""
    from geoparser.db.db import DATABASE_URL

    return Path(DATABASE_URL.replace("sqlite:///", "")).parent


def _resolve_config(args) -> BuildConfig:
    if args.config:
        return BuildConfig.from_yaml(args.config)
    return BuildConfig.for_gazetteer(args.gazetteer)


def _apply_overrides(config: BuildConfig, args) -> BuildConfig:
    """CLI-Werte in die Config ziehen; nicht gesetzte Flags aendern nichts."""
    if args.measure:
        if args.measure not in VALID_MEASURES:
            raise SystemExit(f"Unbekanntes Mass '{args.measure}' "
                             f"(erlaubt: {VALID_MEASURES})")
        config.association_measure = args.measure

    d1_over = {name: getattr(args, name) for name in
               ("r_star", "cap", "k", "d_max", "batch", "threads")
               if getattr(args, name) is not None}
    if args.all_features:
        d1_over["named_only"] = False
    if d1_over:
        config.d1 = replace(config.d1, **d1_over)

    if args.total_area_resolution is not None:
        config.b1 = replace(config.b1,
                            total_area_resolution=args.total_area_resolution)
    return config


def _resolve_db(config: BuildConfig, args) -> Path:
    """Pfad zur H3-DuckDB, aus --db oder aus der Config."""
    db_path = Path(args.db) if args.db is not None else \
        config.resolve_output_path(_geoparser_base_dir())
    if not db_path.exists():
        raise SystemExit(
            f"H3-DuckDB nicht gefunden: {db_path}\n"
            "Pfad mit --db angeben oder zuerst 'spatial-h3-build' ausfuehren."
        )
    return db_path


def _resolve_out_dir(config: BuildConfig, args, db_path: Optional[Path] = None) -> Path:
    """Zielverzeichnis der Matrizen: --out-dir, sonst neben der DuckDB."""
    if args.out_dir:
        return Path(args.out_dir)
    return (db_path or _resolve_db(config, args)).parent


# ─── Rechnen ────────────────────────────────────────────────────────────────────

def run_compute(config: BuildConfig, args) -> Path:
    db_path = _resolve_db(config, args)
    out_dir = _resolve_out_dir(config, args, db_path)
    measure = config.association_measure
    target = out_dir / matrix_filename(measure)

    if target.exists() and not args.overwrite:
        raise SystemExit(
            f"ABBRUCH — {target} ist bereits vorhanden und wird nicht "
            "stillschweigend ueberschrieben.\n"
            "Mit --overwrite erzwingen oder mit --out-dir woanders hinschreiben."
        )

    print("=" * 60)
    print(f"spatial-h3-assoc — Mass {measure}")
    print("=" * 60)
    print(f"  DuckDB: {db_path}")
    print(f"  Ausgabe: {target}")
    if measure == D1:
        p = config.d1
        print(f"  Parameter: r*={p.r_star} cap={p.cap} k={p.k} D={p.d_max} "
              f"batch={p.batch} named_only={p.named_only}")
    else:
        print(f"  Parameter: total_area_resolution="
              f"{config.b1.total_area_resolution}")
    print()

    if measure == D1:
        from ..association.d1 import compute_d1

        compute_d1(db_path, config.d1, output_dir=out_dir,
                   sample_index_path=args.sample_index)
    else:
        from ..association.compute import compute_all

        compute_all(db_path,
                    total_area_resolution=config.b1.total_area_resolution,
                    output_dir=out_dir)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"  sha256 {sha}")

    return target


# ─── Eichung ────────────────────────────────────────────────────────────────────

def _ordered_lists(matrix_path: Path, threshold: float, max_categories: int):
    """{Quellkategorie: [Zielkategorie, ...]} genau so, wie der Generator liest."""
    from ..sentence_generator.association_loader import AssociationMatrixLoader

    loader = AssociationMatrixLoader(matrix_path)
    return {
        cat: [c for c, _ in loader.get_associated_categories(cat, threshold,
                                                             max_categories)]
        for cat in loader.get_all_categories()
    }


def run_report(config: BuildConfig, args) -> None:
    """Kennzahlen je Mass ueber Schwellwert- und Kappungsachse.

    Gerechnet wird ueber denselben Lesepfad wie im Generator
    (AssociationMatrixLoader), damit die Zahlen die tatsaechliche Auswahl
    beschreiben und nicht eine Nachbildung davon.
    """
    out_dir = _resolve_out_dir(config, args)

    vorhanden = [(m, out_dir / matrix_filename(m)) for m in VALID_MEASURES]
    vorhanden = [(m, p) for m, p in vorhanden if p.exists()]
    if not vorhanden:
        raise SystemExit(
            f"Keine Matrix in {out_dir} gefunden "
            f"({', '.join(matrix_filename(m) for m in VALID_MEASURES)})."
        )

    basis_t, basis_c = args.thresholds[0], 10
    print(f"Eichung der Matrizen in {out_dir}")
    print(f"Basis: Schwelle {basis_t}, Kappung {basis_c}\n")

    kopf = (f"{'Mass':<5}{'Schwelle':>10}{'Kappung':>9}{'Zeilen':>8}"
            f"{'Kandidaten/Zeile':>18}{'Listenlaenge':>14}{'gekappt':>9}"
            f"{'Ausschoepfung':>15}{'!= Basis':>10}")
    print(kopf)
    print("-" * len(kopf))

    for measure, path in vorhanden:
        basis = _ordered_lists(path, basis_t, basis_c)
        for threshold in args.thresholds:
            ungekappt = _ordered_lists(path, threshold, 10**9)
            n = len(ungekappt)
            verfuegbar = sum(len(v) for v in ungekappt.values()) / n
            for cap in args.max_categories:
                listen = {k: v[:cap] for k, v in ungekappt.items()}
                laenge = sum(len(v) for v in listen.values()) / n
                gekappt = sum(1 for k in listen if len(ungekappt[k]) > cap)
                anders = sum(1 for k in listen if listen[k] != basis.get(k))
                ausschoepfung = laenge / verfuegbar if verfuegbar else 0.0
                print(f"{measure:<5}{threshold:>10}{cap:>9}{n:>8}"
                      f"{verfuegbar:>18.2f}{laenge:>14.2f}{gekappt:>9}"
                      f"{ausschoepfung:>15.3f}{anders:>10}")
        print()

    print("Lesehilfe: 'Ausschoepfung' ist der Anteil der verfuegbaren Liste, den")
    print("die Kappung durchlaesst; '!= Basis' zaehlt die Zeilen, deren geordnete")
    print("Liste von der Basiszelle desselben Masses abweicht.")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assoziationsmatrizen rechnen (B1 / D1) und eichen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--measure", "-m", choices=list(VALID_MEASURES), default=None,
                        help="Assoziationsmass (Default: association.measure aus der Config)")
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="Pfad zu einer config.yaml")
    parser.add_argument("--gazetteer", "-g", type=str, default="swissnames3d",
                        help="Gazetteer-Name fuer die gebundelte Config")
    parser.add_argument("--db", type=Path, default=None,
                        help="Pfad zur H3-DuckDB (Default: aus der Config)")
    parser.add_argument("--out-dir", "-o", type=Path, default=None,
                        help="Zielverzeichnis der Matrix (Default: neben der DuckDB)")
    parser.add_argument("--overwrite", action="store_true",
                        help="vorhandene Matrix ueberschreiben")

    d1 = parser.add_argument_group("D1 (Adjazenzmass)")
    d1.add_argument("--r-star", type=int, default=None, help="Ankeraufloesung (Default 10)")
    d1.add_argument("--cap", type=int, default=None,
                    help="Stichprobenzellen je Objekt (Default 99999 = faktisch ohne Deckel)")
    d1.add_argument("--k", type=int, default=None, help="Distanzraenge (Default 10)")
    d1.add_argument("--d-max", type=int, default=None,
                    help="Suchradius in Ringschritten (Default 12)")
    d1.add_argument("--batch", type=int, default=None,
                    help="Quellzellen je Rechenblock (Default 40000)")
    d1.add_argument("--threads", type=int, default=None, help="DuckDB-Threads (Default 4)")
    d1.add_argument("--all-features", action="store_true",
                    help="auch Objekte ohne NAME beruecksichtigen (Default: nur benannte)")
    d1.add_argument("--sample-index", type=Path, default=None,
                    help="Parquet mit dem Stichprobenindex: wird gelesen wenn vorhanden, "
                         "sonst dorthin geschrieben")

    b1 = parser.add_argument_group("B1 (Ueberlagerungsmass)")
    b1.add_argument("--total-area-resolution", type=int, default=None,
                    help="Resolution der Gesamtflaeche (Default 10)")

    rep = parser.add_argument_group("Eichung")
    rep.add_argument("--report", action="store_true",
                     help="nur rechnen, was die Matrizen liefern — nichts schreiben")
    rep.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.01, 0.1],
                     help="Schwellwertachse (Default 0.001 0.01 0.1)")
    rep.add_argument("--max-categories", type=int, nargs="+", default=[6, 10, 20],
                     help="Kappungsachse (Default 6 10 20)")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = _apply_overrides(_resolve_config(args), args)
    if args.report:
        run_report(config, args)
    else:
        run_compute(config, args)


if __name__ == "__main__":
    main()
