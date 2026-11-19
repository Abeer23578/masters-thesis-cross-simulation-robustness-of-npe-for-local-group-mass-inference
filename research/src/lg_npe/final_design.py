"""Small, explicit primitives for the final notebook workflows (no model loading)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import load_flamingo_pair_table, load_tng_pair_table
from .features import build_design_matrix
from .splitting import group_split, pair_groups

HISTORICAL_SUM = "out/mass_comparison/stellar_features/m200c_dynamics_stellar_sum/20260917T095930Z"
HISTORICAL_INDIVIDUAL = "out/mass_comparison/stellar_features/m200c_dynamics_stellar_individual/20260917T094931Z"
TARGET_DEFINITION = "log10 common host M200c/Msun if same host; log10 summed host M200c/Msun otherwise"
FINAL_IDENTITY_COLUMNS = (
    'pair_index', 'id_a', 'id_b', 'host_i', 'host_j', 'same_host',
    'separation', 'vr', 'vt', 'mstar_a', 'mstar_b', 'log_m200c',
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def frame_digest(frame):
    """Strict identity of final-six inputs; excludes runtime-derived bound log mass."""
    # Canonical numeric bytes, independent of pandas hashing and host endianness.
    h = hashlib.sha256()
    for c in FINAL_IDENTITY_COLUMNS:
        a = frame[c].to_numpy()
        dtype = '<i8' if a.dtype.kind in 'biu' else '<f8'
        h.update(c.encode()); h.update(np.asarray(a, dtype=dtype).tobytes())
    return h.hexdigest()


def validate_frozen_calibration(project, training):
    """Verify the unchanged calibration through its explicit portability migration.

    The original calibration records its original code/data hashes. Keep that record
    intact; validate relevant training rows plus byte-identical bound-mass sources
    instead of recomputing platform-dependent logarithms for an identity check.
    """
    project = Path(project)
    bundle = project / 'data/derived/final_experiment_v1'
    record = json.loads((bundle / 'calibration_integrity_v3.json').read_text())
    assert record['schema'] == 'frozen-calibration-portability-v3'
    assert record['training_identity'] == {d: frame_digest(f) for d, f in training.items()}, \
        'Frozen calibration training inputs changed'
    for relative, expected in record['file_sha256'].items():
        assert sha256(project / relative) == expected, f'Frozen calibration dependency changed: {relative}'


def load_final_frames(project):
    frames = {}
    for label, loader, expected in [('flamingo', load_flamingo_pair_table, 142713),
                                     ('tng', load_tng_pair_table, 1530)]:
        frame, _, _, _ = build_design_matrix(loader(project).frame, target='m200c',
            feature_block='dynamics', mass_feature='stellar_sum', validity_policy='global_complete_case')
        if len(frame) != expected or not frame.pair_index.is_unique:
            raise ValueError(f'{label}: authoritative population changed')
        np.testing.assert_array_equal(frame.same_host, frame.host_i == frame.host_j)
        frames[label] = frame
    return frames


def assert_partition(frame, parts):
    arrays = [np.asarray(parts[k], dtype=int) for k in ('train', 'validation', 'test')]
    joined = np.concatenate(arrays)
    if len(joined) != len(frame) or not np.array_equal(np.sort(joined), np.arange(len(frame))):
        raise ValueError('Split must be an exhaustive disjoint partition')
    groups = pair_groups(frame)
    for i in range(3):
        for j in range(i):
            if np.intersect1d(groups[arrays[i]], groups[arrays[j]]).size:
                raise ValueError('Connected-host leakage')


def final_splits(project, frames):
    result = {}
    for domain, frame in frames.items():
        outer_train, test, groups = group_split(frame, test_size=.15, seed=42)
        if domain == 'flamingo':
            with np.load(Path(project) / HISTORICAL_SUM / 'split_indices.npz') as saved:
                np.testing.assert_array_equal(outer_train, saved['train'])
                np.testing.assert_array_equal(test, saved['test'])
                np.testing.assert_array_equal(groups, saved['source_groups'])
        a, b, _ = group_split(frame.iloc[outer_train], test_size=.1, seed=43)
        result[domain] = dict(train=outer_train[a], validation=outer_train[b], test=test, groups=groups)
        assert_partition(frame, result[domain])
    return result


def equal_domain_weights(domain):
    domain = np.asarray(domain)
    labels, inverse, counts = np.unique(domain, return_inverse=True, return_counts=True)
    if not len(labels):
        raise ValueError('Empty training population')
    return 1.0 / (len(labels) * counts[inverse])


def fit_training_scaler(X, domain):
    return StandardScaler().fit(np.asarray(X, float), sample_weight=equal_domain_weights(domain))


def balanced_batch_indices(domain, batch_size, rng):
    domain = np.asarray(domain)
    labels = np.unique(domain)
    if not len(labels) or batch_size % len(labels):
        raise ValueError('Batch size must be divisible by the number of domains')
    batch = np.concatenate([rng.choice(np.flatnonzero(domain == d),
        batch_size // len(labels), replace=True) for d in labels])
    return rng.permutation(batch)


def shift_stellar_features(X, names, delta):
    """Add dex in raw feature space. Dynamics and ratios are exact invariants."""
    result = np.asarray(X, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != len(names) or not np.isfinite(delta):
        raise ValueError('Invalid feature matrix or Delta')
    for i, name in enumerate(names):
        if name in {'log_stellar_sum', 'log_mstar_primary', 'log_mstar_secondary'}:
            result[:, i] += delta
    return result


def component_weights(groups, rng):
    _, inverse = np.unique(groups, return_inverse=True)
    n = inverse.max() + 1
    return np.bincount(rng.integers(n, size=n), minlength=n)[inverse]


def calibrate_delta(train_frames, repeats=500, seed=261001):
    """Forward bound-mass mapping; caller passes training rows ONLY.

    Does not read M200c or NPE errors. Reference is fixed across bootstrap draws.
    """
    columns = ['separation', 'vr', 'vt', 'log_stellar_sum', 'same_host']
    f, t = [train_frames[d] for d in ('flamingo', 'tng')]
    xf, xt = [d[columns].to_numpy(float) for d in (f, t)]
    centre = (xf.mean(0) + xt.mean(0)) / 2
    scale = np.sqrt((((xf-centre)**2).mean(0) + ((xt-centre)**2).mean(0))/2)
    if np.any(scale <= 0):
        raise ValueError('Degenerate calibration design')
    design = [np.column_stack([np.ones(len(x)), (x-centre)/scale]) for x in (xf, xt)]
    outcomes = [d.log_bound_subhalo_sum.to_numpy(float) for d in (f, t)]

    def fit(weights):
        betas = [np.linalg.lstsq(x*np.sqrt(w[:,None]), y*np.sqrt(w), rcond=None)[0]
                 for x, y, w in zip(design, outcomes, weights)]
        gap = betas[1][0] - betas[0][0]
        slope = betas[0][4] / scale[3]
        if slope <= .05:
            raise ValueError('Calibration stellar slope is not reliably positive')
        return gap / slope, gap, slope

    point, gap, slope = fit([np.ones(len(f)), np.ones(len(t))])
    rng = np.random.default_rng(seed)
    groups = [pair_groups(d) for d in (f, t)]
    draws = np.array([fit([component_weights(g, rng) for g in groups])[0]
                      for _ in range(repeats)])
    return dict(delta=float(point), bound_gap=float(gap), flamingo_stellar_slope=float(slope),
        interval95=np.quantile(draws, [.025,.975]).tolist(), predictors=columns,
        reference_centre=centre.tolist(), reference_scale=scale.tolist(),
        n_flamingo=len(f), n_tng=len(t), seed=seed, repeats=repeats,
        definition='training-only bound-mass intercept gap / FLAMINGO stellar slope',
        input_role='calibration independent of held-out target and NPE performance'), draws


def posterior_summary(samples, truth, pair_index, groups, same_host):
    q = np.percentile(samples, [2.5,16,50,84,97.5], axis=0)
    sd = samples.std(axis=0)
    return pd.DataFrame(dict(pair_index=pair_index, group=groups, same_host=same_host,
        truth=truth, q025=q[0], q16=q[1], median=q[2], q84=q[3], q975=q[4],
        std=sd, rank=(samples < np.asarray(truth)[None,:]).sum(0), draws=samples.shape[0]))


def summary_metrics(frame, weights=None):
    if weights is None:
        weights = np.ones(len(frame))
    w = np.asarray(weights, float)
    if len(frame) == 0 or w.sum() == 0:
        return None
    w = w / w.sum()
    y, med = frame.truth.to_numpy(), frame['median'].to_numpy()
    residual = med-y
    variance = np.sum(w*(y-np.sum(w*y))**2)
    n_draws = int(frame.draws.iloc[0])
    rank_cdf = np.bincount(frame['rank'].to_numpy(int), weights=w, minlength=n_draws+1).cumsum()
    return dict(rmse=float(np.sqrt(np.sum(w*residual**2))), bias=float(np.sum(w*residual)),
        r2=float(1-np.sum(w*residual**2)/variance) if variance > 0 else np.nan,
        coverage68=float(np.sum(w*((y>=frame.q16)&(y<=frame.q84)))),
        coverage95=float(np.sum(w*((y>=frame.q025)&(y<=frame.q975)))),
        mean_width68=float(np.sum(w*(frame.q84-frame.q16))),
        mean_width95=float(np.sum(w*(frame.q975-frame.q025))),
        rank_cdf_distance=float(np.max(np.abs(rank_cdf-np.arange(1,n_draws+2)/(n_draws+1)))))


def paired_component_intervals(a, b, repeats=2000, seed=261001):
    """B minus A. Cluster sampling before host stratification preserves dependence."""
    for col in ['pair_index','truth','group','same_host']:
        np.testing.assert_array_equal(a[col], b[col])
    rng = np.random.default_rng(seed)
    rows = []
    masks = dict(pooled=np.ones(len(a), bool), same_host=a.same_host.to_numpy(bool),
                 different_host=~a.same_host.to_numpy(bool))
    for repeat in range(repeats):
        w = component_weights(a.group, rng)
        for subset, mask in masks.items():
            ma, mb = summary_metrics(a.loc[mask], w[mask]), summary_metrics(b.loc[mask], w[mask])
            if ma is None:
                continue
            for metric in ma:
                rows.append(dict(repeat=repeat, subset=subset, metric=metric, value=mb[metric]-ma[metric]))
            rows.append(dict(repeat=repeat, subset=subset, metric='absolute_bias',
                             value=abs(mb['bias'])-abs(ma['bias'])))
    raw = pd.DataFrame(rows)
    summary = raw.groupby(['subset','metric']).value.agg(
        median='median', lower95=lambda x:x.quantile(.025), upper95=lambda x:x.quantile(.975),
        repeats='size').reset_index()
    return summary, raw


def require_saved_model_environment():
    import importlib.metadata as m
    import platform
    expected = {'sbi':'0.27.0', 'torch':'2.13.0', 'scikit-learn':'1.9.0',
                'numpy':'2.5.2', 'scipy':'1.18.1'}
    actual = {k:m.version(k).split('+')[0] for k in expected}
    errors = {k:(actual[k],v) for k,v in expected.items() if actual[k] != v}
    if platform.python_version() != '3.12.14':
        errors['python'] = (platform.python_version(), '3.12.14')
    if errors:
        raise RuntimeError(f'Use pinned Roihu environment before loading/training: {errors}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Roihu CUDA required')


def load_model(path):
    require_saved_model_environment()
    import pickle
    with Path(path).open('rb') as f:
        return pickle.load(f)
