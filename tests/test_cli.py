import json

from psm_fdr_audit.cli import main


def test_simulate_and_audit_roundtrip(tmp_path):
    table_path = tmp_path / "psms.csv"
    report_path = tmp_path / "report.json"
    assert main(["simulate", "--scenario", "ties", "--output", str(table_path)]) == 0
    assert (
        main(
            [
                "audit",
                str(table_path),
                "--false-column",
                "is_false",
                "--entrapment-column",
                "is_entrapment",
                "--entrapment-ratio",
                "0.1",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [row["threshold"] for row in report] == [0.001, 0.01, 0.05]
