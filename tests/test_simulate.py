import numpy as np

from psm_fdr_audit.core import audit_thresholds, target_decoy_q_values
from psm_fdr_audit.simulate import simulate_psms


def test_simulation_is_reproducible():
    first = simulate_psms(seed=7)
    second = simulate_psms(seed=7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_tie_scenario_contains_repeated_scores():
    simulation = simulate_psms(scenario="ties", seed=5)
    assert np.unique(simulation["score"]).size < simulation["score"].size


def test_decoy_shift_exposes_anti_conservative_behavior():
    simulation = simulate_psms(
        scenario="decoy_shift",
        n_true=8_000,
        n_false_target=4_000,
        n_decoy=4_000,
        n_entrapment=0,
        seed=17,
    )
    q_values = target_decoy_q_values(simulation["score"], simulation["is_decoy"])
    report = audit_thresholds(
        q_values,
        simulation["is_decoy"],
        thresholds=[0.05],
        is_false=simulation["is_false"],
    )[0]
    assert report["target_discoveries"] > 0
    assert report["realized_fdp"] > 0.05
