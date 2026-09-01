# psm-fdr-audit

`psm-fdr-audit` is a small Python package for inspecting target–decoy FDR calculations on peptide-spectrum match (PSM) scores. It focuses on transparent calculations, deterministic synthetic examples, score-tie handling, and failure scenarios that can be checked end to end.

This is an educational and testing tool. It is **not** a replacement for a validated search-engine workflow, [mokapot](https://github.com/wfondrie/mokapot), or the database-generation and FDP-estimation methods in [FDRBench](https://github.com/Noble-Lab/FDRBench).

## Why this project exists

Small implementation choices can change reported discoveries:

- tied scores can produce row-order-dependent results if they are split;
- zero observed decoys can produce an estimated FDR of zero without a correction;
- decoys that do not resemble incorrect targets can make target–decoy estimates anti-conservative;
- truth labels available in simulations can reveal gaps between an estimated FDR and realized FDP.

The project turns these cases into executable tests rather than treating a plausible q-value column as sufficient evidence of calibration.

## Included

- tie-aware target–decoy q-values;
- configurable score direction and decoy pseudocount;
- threshold-level discovery summaries;
- optional realized FDP when simulation truth is available;
- an explicitly labelled scaled entrapment-rate diagnostic;
- calibrated, tied-score, and shifted-decoy simulations;
- CSV and JSON command-line workflow;
- unit tests and multi-version GitHub Actions CI.

## Quick start

```bash
python -m pip install -e ".[test]"

psm-fdr-audit simulate \
  --scenario ties \
  --output audit-output/tied-psms.csv

psm-fdr-audit audit audit-output/tied-psms.csv \
  --false-column is_false \
  --entrapment-column is_entrapment \
  --entrapment-ratio 0.1 \
  --output audit-output/tied-report.json
```

The `decoy_shift` scenario deliberately gives decoys a lower score distribution than incorrect target PSMs. It is meant to demonstrate a failure mode, not to represent a particular instrument, search engine, or dataset.

## Python API

```python
from psm_fdr_audit import audit_thresholds, simulate_psms, target_decoy_q_values

simulation = simulate_psms(scenario="decoy_shift", seed=13)
q_values = target_decoy_q_values(
    simulation["score"],
    simulation["is_decoy"],
)
report = audit_thresholds(
    q_values,
    simulation["is_decoy"],
    is_false=simulation["is_false"],
    is_entrapment=simulation["is_entrapment"],
    entrapment_ratio=0.1,
)
```

## Interpretation boundaries

- The simulation distributions are intentionally simple and do not model a specific DIA or DDA pipeline.
- The scaled entrapment rate is reported transparently and should not be interpreted as a universally valid FDP estimator. Entrapment scaling depends on database construction, competition rules, level of inference, and shared sequences.
- The implementation operates on supplied top-ranked PSM scores; it does not perform spectrum matching, candidate competition, peptide roll-up, or protein inference.
- Results should be compared with the assumptions of the actual search and evaluation workflow.

## Development

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## License

MIT
