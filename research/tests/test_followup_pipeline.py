import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lg_npe.config import load_experiment_config
from lg_npe.constants import HISTORICAL_NPE_DEFAULTS
from lg_npe.evaluation import posterior_metrics
from lg_npe.features import build_design_matrix, ordered_stellar_masses
from lg_npe.splitting import group_split
from lg_npe.validation import validate_experiment_config, validate_no_bound_mass_leakage


ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame():
    return pd.DataFrame({
        "host_i": [1, 1, 3, 5], "host_j": [2, 2, 4, 6],
        "id_a": [10, 11, 12, 13], "id_b": [20, 21, 22, 23],
        "mstar_a": [10., 20., 30., 40.], "mstar_b": [20., 10., 30., 50.],
        "separation": [600., 700., 800., 900.], "vr": [-100., -110., -120., -130.],
        "vt": [50., 60., 70., 80.], "log_mass_ratio": [-.3, -.4, -.5, -.6],
        "sigma_mean": [100., 101., np.nan, 103.], "sigma_diff": [5., 6., 7., 8.],
        "kappa_mean": [.4, .5, .6, .7], "kappa_diff": [.1, .1, .2, .2],
        "met_diff": [.01, .02, .03, .04], "gr_mean": [.5, .6, .7, .8],
        "gr_diff": [.1, .2, .3, .4], "log_m200c": [12., 12.1, 12.2, 12.3],
        "log_bound_subhalo_sum": [11.8, 11.9, 12.0, 12.1],
    })


def test_stellar_ordering_and_stable_tie_break():
    primary, secondary, a_primary = ordered_stellar_masses(
        np.array([1., 3., 2.]), np.array([2., 1., 2.]),
        np.array([9, 9, 5]), np.array([8, 8, 7]),
    )
    np.testing.assert_array_equal(primary, [2., 3., 2.])
    np.testing.assert_array_equal(secondary, [1., 1., 2.])
    np.testing.assert_array_equal(a_primary, [False, True, True])


def test_global_complete_case_is_shared_across_mass_features():
    frame = synthetic_frame()
    populations = []
    for mass_feature in ("none", "bound_subhalo_ratio", "stellar_individual", "stellar_ratio", "stellar_sum"):
        selected, _, _, counts = build_design_matrix(
            frame, target="m200c", feature_block="dynamics", mass_feature=mass_feature,
            validity_policy="global_complete_case",
        )
        populations.append(selected.index.to_list())
        assert counts["final_pairs"] == 3
    assert all(population == populations[0] for population in populations)


def test_per_configuration_policy_is_available_but_not_default():
    selected, _, _, counts = build_design_matrix(
        synthetic_frame(), target="m200c", feature_block="dynamics", mass_feature="none",
        validity_policy="per_configuration",
    )
    assert len(selected) == 4
    assert counts["final_pairs"] == 4


def test_bound_target_rejects_bound_ratio():
    config = {
        "simulation_train": "flamingo", "simulation_test": "tng",
        "target": "bound_subhalo_sum", "feature_block": "dynamics",
        "mass_feature": "bound_subhalo_ratio", "validity_policy": "global_complete_case",
        "split": {"method": "connected_host_components"}, "scaler": {"fit_on": "source_train"},
        "npe": {},
    }
    with pytest.raises(ValueError, match="Leakage"):
        validate_experiment_config(config)
    with pytest.raises(ValueError, match="Leakage"):
        validate_no_bound_mass_leakage("bound_subhalo_sum", ["separation", "Msub_bound_1"])


def test_saved_num_layers_is_not_accepted():
    config = {
        "simulation_train": "flamingo", "simulation_test": "tng",
        "target": "m200c", "feature_block": "dynamics", "mass_feature": "none",
        "validity_policy": "global_complete_case",
        "split": {"method": "connected_host_components"}, "scaler": {"fit_on": "source_train"},
        "npe": {"num_layers": 2},
    }
    with pytest.raises(ValueError, match="num_layers"):
        validate_experiment_config(config)
    assert "num_layers" not in HISTORICAL_NPE_DEFAULTS


def test_group_split_has_disjoint_connected_components():
    frame = pd.DataFrame({
        "host_i": [1, 2, 10, 11, 20, 21, 30, 31],
        "host_j": [2, 3, 11, 12, 21, 22, 31, 32],
    })
    train, test, groups = group_split(frame, test_size=.25, seed=42)
    assert not (set(groups[train]) & set(groups[test]))


def test_all_committed_configs_validate():
    paths = sorted((ROOT / "configs/npe").rglob("*.json"))
    assert len(paths) == 10
    for path in paths:
        config = load_experiment_config(path)
        assert config["validity_policy"] == "global_complete_case"
        assert "num_layers" not in config["npe"]


def test_posterior_metrics_include_calibration_and_sharpness():
    samples = np.array([
        [0.8, 1.8, 2.8], [0.9, 1.9, 2.9], [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1], [1.2, 2.2, 3.2],
    ])
    result = posterior_metrics(samples, np.array([1., 2., 3.]))
    for key in ("rmse", "bias", "r2", "coverage68", "coverage95", "median_beta",
                "rank_cdf_distance", "mean_width68", "mean_width95"):
        assert key in result
    assert result["rmse"] == pytest.approx(0.0)
