"""Command-line interface for synthetic and tabular FDR audits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from psm_fdr_audit.core import audit_thresholds, target_decoy_q_values
from psm_fdr_audit.simulate import simulate_psms


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _write_simulation(path: Path, simulation: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["score", "label", "is_decoy", "is_entrapment", "is_false"])
        writer.writerows(
            zip(
                simulation["score"],
                simulation["label"],
                simulation["is_decoy"],
                simulation["is_entrapment"],
                simulation["is_false"],
                strict=True,
            )
        )


def _read_columns(
    path: Path,
    *,
    score_column: str,
    decoy_column: str,
    false_column: str | None,
    entrapment_column: str | None,
) -> dict[str, np.ndarray | None]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input table has no data rows")
    return {
        "score": np.array([float(row[score_column]) for row in rows]),
        "is_decoy": np.array([_parse_bool(row[decoy_column]) for row in rows]),
        "is_false": (
            None
            if false_column is None
            else np.array([_parse_bool(row[false_column]) for row in rows])
        ),
        "is_entrapment": (
            None
            if entrapment_column is None
            else np.array([_parse_bool(row[entrapment_column]) for row in rows])
        ),
    }


def _audit(args: argparse.Namespace) -> list[dict[str, float | int | None]]:
    columns = _read_columns(
        args.input,
        score_column=args.score_column,
        decoy_column=args.decoy_column,
        false_column=args.false_column,
        entrapment_column=args.entrapment_column,
    )
    q_values = target_decoy_q_values(
        columns["score"],
        columns["is_decoy"],
        higher_better=not args.lower_better,
        pseudocount=args.pseudocount,
    )
    return audit_thresholds(
        q_values,
        columns["is_decoy"],
        thresholds=args.threshold,
        is_false=columns["is_false"],
        is_entrapment=columns["is_entrapment"],
        entrapment_ratio=args.entrapment_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psm-fdr-audit",
        description="Run small, auditable target-decoy FDR experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser(
        "simulate", help="write a synthetic PSM table"
    )
    simulate_parser.add_argument(
        "--scenario",
        choices=["calibrated", "ties", "decoy_shift"],
        default="calibrated",
    )
    simulate_parser.add_argument("--seed", type=int, default=13)
    simulate_parser.add_argument("--output", type=Path, required=True)

    audit_parser = subparsers.add_parser("audit", help="audit a CSV PSM table")
    audit_parser.add_argument("input", type=Path)
    audit_parser.add_argument("--score-column", default="score")
    audit_parser.add_argument("--decoy-column", default="is_decoy")
    audit_parser.add_argument("--false-column")
    audit_parser.add_argument("--entrapment-column")
    audit_parser.add_argument("--entrapment-ratio", type=float)
    audit_parser.add_argument("--lower-better", action="store_true")
    audit_parser.add_argument("--pseudocount", type=float, default=1.0)
    audit_parser.add_argument("--threshold", type=float, action="append", default=None)
    audit_parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        _write_simulation(
            args.output,
            simulate_psms(scenario=args.scenario, seed=args.seed),
        )
        return 0

    if args.threshold is None:
        args.threshold = [0.001, 0.01, 0.05]
    report = _audit(args)
    rendered = json.dumps(report, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
