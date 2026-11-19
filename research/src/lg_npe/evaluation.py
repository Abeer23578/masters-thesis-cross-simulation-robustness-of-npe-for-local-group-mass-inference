from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def posterior_metrics(samples: np.ndarray, truth: np.ndarray) -> dict:
    samples = np.asarray(samples)
    truth = np.asarray(truth)
    if samples.ndim != 2 or samples.shape[1] != len(truth):
        raise ValueError("samples must have shape (draws, observations)")
    med = np.median(samples, axis=0)
    q025, q16, q84, q975 = np.percentile(samples, [2.5, 16, 84, 97.5], axis=0)
    residual = med - truth
    std = samples.std(axis=0)
    beta = np.divide(np.abs(residual), std, out=np.full_like(std, np.inf), where=std > 0)
    ranks = (samples < truth[None, :]).sum(axis=0)
    rank_counts = np.bincount(ranks, minlength=samples.shape[0] + 1)
    rank_reference_cdf = np.arange(1, samples.shape[0] + 2) / (samples.shape[0] + 1)
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(truth, med)) if len(truth) > 1 else None,
        "coverage68": float(np.mean((truth >= q16) & (truth <= q84))),
        "coverage95": float(np.mean((truth >= q025) & (truth <= q975))),
        "median_beta": float(np.median(beta)),
        "rank_cdf_distance": float(
            np.max(np.abs(rank_counts.cumsum() / len(truth) - rank_reference_cdf))
        ),
        "mean_width68": float(np.mean(q84 - q16)),
        "mean_width95": float(np.mean(q975 - q025)),
    }
