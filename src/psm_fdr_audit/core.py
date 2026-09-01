"""Core target-decoy calculations and compact audit summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


def _validate_scores_and_labels(
    scores: npt.ArrayLike, is_decoy: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=float)
    decoy_array = np.asarray(is_decoy, dtype=bool)
    if score_array.ndim != 1 or decoy_array.ndim != 1:
        raise ValueError("scores and is_decoy must be one-dimensional")
    if score_array.size != decoy_array.size:
        raise ValueError("scores and is_decoy must have equal lengths")
    if score_array.size == 0:
        raise ValueError("at least one score is required")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values")
    return score_array, decoy_array


def target_decoy_q_values(
    scores: npt.ArrayLike,
    is_decoy: npt.ArrayLike,
    *,
    higher_better: bool = True,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Estimate q-values with score-tie-aware target-decoy competition.

    All PSMs with an identical score receive the same estimate. This avoids
    row-order-dependent results when tied scores cross a reporting threshold.

    Parameters
    ----------
    scores
        PSM scores.
    is_decoy
        Boolean labels where ``True`` denotes a decoy PSM.
    higher_better
        Whether larger scores rank ahead of smaller scores.
    pseudocount
        Value added to the cumulative decoy count. A value of one is a common
        conservative choice; zero reproduces the uncorrected ratio.
    """
    score_array, decoy_array = _validate_scores_and_labels(scores, is_decoy)
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")

    order = np.argsort(-score_array if higher_better else score_array, kind="stable")
    ranked_scores = score_array[order]
    ranked_decoys = decoy_array[order]

    group_end = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    group_end_indices = np.flatnonzero(group_end)
    cumulative_decoys = np.cumsum(ranked_decoys)[group_end_indices]
    cumulative_targets = np.cumsum(~ranked_decoys)[group_end_indices]
    fdr_by_group = np.minimum(
        1.0,
        (cumulative_decoys + pseudocount) / np.maximum(cumulative_targets, 1),
    )
    q_by_group = np.minimum.accumulate(fdr_by_group[::-1])[::-1]

    ranked_q_values = np.empty(score_array.size, dtype=float)
    group_start = 0
    for group_stop, q_value in zip(group_end_indices + 1, q_by_group, strict=True):
        ranked_q_values[group_start:group_stop] = q_value
        group_start = group_stop

    q_values = np.empty_like(ranked_q_values)
    q_values[order] = ranked_q_values
    return q_values


def audit_thresholds(
    q_values: npt.ArrayLike,
    is_decoy: npt.ArrayLike,
    *,
    thresholds: Iterable[float] = (0.001, 0.01, 0.05),
    is_false: npt.ArrayLike | None = None,
    is_entrapment: npt.ArrayLike | None = None,
    entrapment_ratio: float | None = None,
) -> list[dict[str, float | int | None]]:
    """Summarize discoveries and optional truth/entrapment diagnostics.

    ``scaled_entrapment_rate`` is a transparent diagnostic rather than a
    general-purpose FDP estimator. It is calculated as the accepted
    entrapment fraction divided by ``entrapment_ratio``. The correct scaling
    depends on database construction and competition design.
    """
    q_array = np.asarray(q_values, dtype=float)
    decoy_array = np.asarray(is_decoy, dtype=bool)
    if q_array.ndim != 1 or decoy_array.ndim != 1:
        raise ValueError("q_values and is_decoy must be one-dimensional")
    if q_array.size != decoy_array.size:
        raise ValueError("q_values and is_decoy must have equal lengths")

    false_array = None if is_false is None else np.asarray(is_false, dtype=bool)
    entrapment_array = (
        None if is_entrapment is None else np.asarray(is_entrapment, dtype=bool)
    )
    for name, optional_array in (
        ("is_false", false_array),
        ("is_entrapment", entrapment_array),
    ):
        if optional_array is not None and optional_array.shape != q_array.shape:
            raise ValueError(f"{name} must have the same shape as q_values")
    if entrapment_array is not None and (
        entrapment_ratio is None or entrapment_ratio <= 0
    ):
        raise ValueError("a positive entrapment_ratio is required with is_entrapment")

    rows: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("thresholds must be between zero and one")
        accepted = q_array <= threshold
        accepted_targets = accepted & ~decoy_array
        target_count = int(np.sum(accepted_targets))
        decoy_count = int(np.sum(accepted & decoy_array))
        realized_fdp = (
            None
            if false_array is None or target_count == 0
            else float(np.sum(accepted_targets & false_array) / target_count)
        )
        scaled_entrapment_rate = (
            None
            if entrapment_array is None or target_count == 0
            else float(
                np.sum(accepted_targets & entrapment_array)
                / target_count
                / entrapment_ratio
            )
        )
        rows.append(
            {
                "threshold": float(threshold),
                "target_discoveries": target_count,
                "decoy_discoveries": decoy_count,
                "realized_fdp": realized_fdp,
                "scaled_entrapment_rate": scaled_entrapment_rate,
            }
        )
    return rows
