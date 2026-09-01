"""Synthetic PSM score generators with explicit ground truth."""

from __future__ import annotations

import numpy as np


def simulate_psms(
    *,
    scenario: str = "calibrated",
    n_true: int = 4_000,
    n_false_target: int = 1_000,
    n_decoy: int = 1_000,
    n_entrapment: int = 500,
    seed: int = 13,
) -> dict[str, np.ndarray]:
    """Generate synthetic scores for a small FDR-control stress test.

    The simulation is intentionally simple. It is useful for checking code
    paths and qualitative failure modes, not for modelling a specific search
    engine or experiment.
    """
    counts = (n_true, n_false_target, n_decoy, n_entrapment)
    if any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError("simulation counts must be non-negative and not all zero")
    if scenario not in {"calibrated", "ties", "decoy_shift"}:
        raise ValueError("scenario must be calibrated, ties, or decoy_shift")

    rng = np.random.default_rng(seed)
    true_scores = rng.normal(3.0, 1.0, n_true)
    false_scores = rng.normal(0.0, 1.0, n_false_target)
    decoy_location = -0.6 if scenario == "decoy_shift" else 0.0
    decoy_scores = rng.normal(decoy_location, 1.0, n_decoy)
    entrapment_scores = rng.normal(0.0, 1.0, n_entrapment)

    scores = np.concatenate(
        [true_scores, false_scores, decoy_scores, entrapment_scores]
    )
    labels = np.concatenate(
        [
            np.full(n_true, "true_target", dtype=object),
            np.full(n_false_target, "false_target", dtype=object),
            np.full(n_decoy, "decoy", dtype=object),
            np.full(n_entrapment, "entrapment", dtype=object),
        ]
    )
    if scenario == "ties":
        scores = np.round(scores, 1)

    permutation = rng.permutation(scores.size)
    labels = labels[permutation]
    return {
        "score": scores[permutation],
        "label": labels,
        "is_decoy": labels == "decoy",
        "is_entrapment": labels == "entrapment",
        "is_false": labels != "true_target",
    }
