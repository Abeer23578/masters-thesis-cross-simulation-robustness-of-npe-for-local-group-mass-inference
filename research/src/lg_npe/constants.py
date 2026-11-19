from __future__ import annotations

BASE_DYNAMICS = ["separation", "vr", "vt"]

BLOCK_ADDITIONS = {
    "dynamics": [],
    "chemistry": ["met_diff"],
    "morphology": ["sigma_mean", "sigma_diff", "kappa_mean", "kappa_diff"],
    "photometry": ["gr_mean", "gr_diff"],
    "all": [
        "sigma_mean", "sigma_diff", "kappa_mean", "kappa_diff",
        "met_diff", "gr_mean", "gr_diff",
    ],
}

MASS_FEATURE_COLUMNS = {
    "none": [],
    "bound_subhalo_ratio": ["log_mass_ratio"],
    "stellar_individual": ["log_mstar_primary", "log_mstar_secondary"],
    "stellar_ratio": ["log_stellar_ratio"],
    "stellar_sum": ["log_stellar_sum"],
}

TARGET_COLUMNS = {
    "m200c": "log_m200c",
    "bound_subhalo_sum": "log_bound_subhalo_sum",
}

# This reproduces the complete-case population used by the historical baseline.
BASELINE_COMPLETE_CASE_FEATURES = [
    "separation", "vr", "vt", "log_mass_ratio",
    "sigma_mean", "sigma_diff", "kappa_mean", "kappa_diff",
    "met_diff", "gr_mean", "gr_diff",
]

BOUND_MASS_DERIVED_FEATURES = {
    "Msub_bound_1", "Msub_bound_2", "Msub_bound_sum",
    "log_msub_bound_1", "log_msub_bound_2", "log_bound_subhalo_sum",
    "log_mass_ratio", "bound_subhalo_ratio", "bound_subhalo_sum",
}

HISTORICAL_NPE_DEFAULTS = {
    "model": "maf",
    "hidden_features": 64,
    "num_transforms": 5,
    "learning_rate": 0.00016488693479340492,
    "batch_size": 64,
    "max_num_epochs": 200,
    "stop_after_epochs": 20,
    "prior_low": 11.5,
    "prior_high": 14.5,
    # Deliberately no num_layers: the historical posterior_nn call did not use it.
}
