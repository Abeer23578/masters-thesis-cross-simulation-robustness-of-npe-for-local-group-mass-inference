"""Observation integration algebra; no invented Local Group likelihood."""
import numpy as np
from scipy.special import logsumexp


def normalized_importance_weights(log_likelihood, log_population, log_proposal):
    terms = [np.asarray(x, float) for x in (log_likelihood, log_population, log_proposal)]
    if any(x.shape != terms[0].shape for x in terms) or terms[0].ndim != 1:
        raise ValueError('Log weights must be aligned vectors')
    logw = terms[0] + terms[1] - terms[2]
    if np.any(np.isnan(logw)) or np.any(np.isposinf(logw)) or not np.isfinite(logw).any():
        raise ValueError('Invalid or zero-support importance weights')
    w = np.exp(logw-logsumexp(logw))
    return w, float(1/np.sum(w*w))


def gaussian_linear_observation(prior_variance, noise_variance, y):
    """Analytic x|y for synthetic x~N(0,v), y|x~N(x,s2)."""
    if min(prior_variance, noise_variance) <= 0:
        raise ValueError('Variances must be positive')
    variance = 1/(1/prior_variance+1/noise_variance)
    return variance*y/noise_variance, variance
