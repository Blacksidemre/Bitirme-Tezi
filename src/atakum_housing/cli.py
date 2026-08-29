"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atakum konut ilanları için tekrarlanabilir analiz hattı"
    )
    parser.add_argument("--data", default="veriseti.xlsx", help="Girdi Excel dosyası")
    parser.add_argument("--output-dir", default="outputs/latest", help="Üretilen çıktı klasörü")
    parser.add_argument(
        "--reported-raw-rows",
        type=int,
        default=2_836,
        help="Tezde bildirilen ilk ham örneklem büyüklüğü",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        reported_raw_rows=args.reported_raw_rows,
        random_state=args.random_state,
    )
    best = summary["best_model"]
    audit = summary["data_audit"]
    print(
        json.dumps(
            {
                "status": "ok",
                "repository_rows": audit["repository_rows"],
                "latest_unique_listings": audit["latest_snapshot_rows"],
                "best_model": best["name"],
                "test_mae": best["locked_test"]["test_mae"],
                "test_r2": best["locked_test"]["test_r2"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
