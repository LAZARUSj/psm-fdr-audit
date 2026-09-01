import numpy as np
import pytest

from psm_fdr_audit.core import audit_thresholds, target_decoy_q_values


def test_q_values_are_monotone_as_score_threshold_relaxes():
    scores = np.array([10, 9, 8, 7, 6, 5], dtype=float)
    is_decoy = np.array([False, False, True, False, True, False])
    q_values = target_decoy_q_values(scores, is_decoy)
    ranked_q_values = q_values[np.argsort(-scores)]
    assert np.all(np.diff(ranked_q_values) >= 0)


def test_tied_scores_are_invariant_to_row_order():
    scores = np.array([5, 5, 5, 4, 4, 3], dtype=float)
    is_decoy = np.array([False, True, False, True, False, False])
    q_values = target_decoy_q_values(scores, is_decoy)

    permutation = np.array([2, 0, 1, 4, 3, 5])
    permuted = target_decoy_q_values(scores[permutation], is_decoy[permutation])
    restored = np.empty_like(permuted)
    restored[permutation] = permuted

    np.testing.assert_allclose(q_values, restored)
    assert len(set(q_values[scores == 5])) == 1
    assert len(set(q_values[scores == 4])) == 1


def test_pseudocount_prevents_zero_estimate_without_decoys():
    scores = np.array([3, 2, 1], dtype=float)
    is_decoy = np.zeros(3, dtype=bool)
    corrected = target_decoy_q_values(scores, is_decoy, pseudocount=1)
    uncorrected = target_decoy_q_values(scores, is_decoy, pseudocount=0)
    assert np.all(corrected > 0)
    assert np.all(uncorrected == 0)


def test_audit_reports_truth_and_scaled_entrapment_rate():
    q_values = np.array([0.005, 0.005, 0.02, 0.005])
    is_decoy = np.array([False, False, True, False])
    is_false = np.array([False, True, True, True])
    is_entrapment = np.array([False, False, False, True])
    report = audit_thresholds(
        q_values,
        is_decoy,
        thresholds=[0.01],
        is_false=is_false,
        is_entrapment=is_entrapment,
        entrapment_ratio=0.5,
    )[0]
    assert report["target_discoveries"] == 3
    assert report["realized_fdp"] == pytest.approx(2 / 3)
    assert report["scaled_entrapment_rate"] == pytest.approx(2 / 3)
