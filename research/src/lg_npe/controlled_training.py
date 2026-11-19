"""Opt-in supervised NPE training with explicit validation and domain loss.

Historical train_npe is deliberately untouched. No proposal/prior reweighting.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .constants import HISTORICAL_NPE_DEFAULTS
from .final_design import fit_training_scaler


def draw_batch_from_pools(pools, batch_size, rng):
    """Cached domain indices avoid rescanning 100k rows at every GPU update."""
    if not pools or batch_size % len(pools):
        raise ValueError('Batch size must be divisible by the number of domains')
    return rng.permutation(np.concatenate([
        rng.choice(pool,batch_size//len(pools),replace=True) for pool in pools]))


def validate_final_config(config):
    if config['training_domains'] not in [['flamingo'], ['tng'], ['flamingo','tng']]:
        raise ValueError('Unsupported training domains or order')
    if config['mass_feature'] not in ('none','stellar_sum') or config['target'] != 'm200c':
        raise ValueError('Final matrix is fixed to dynamics / stellar sum and M200c')
    if config['npe'] != HISTORICAL_NPE_DEFAULTS:
        raise ValueError('Architecture/training settings changed without a decision revision')
    if config['observation_noise'] != 'none' or config['objective'] != 'equal_domain_log_loss':
        raise ValueError('Final objective/noise policy changed')
    if config['posterior_draws'] != 500 or config['seed'] != 42:
        raise ValueError('Final evaluation/seed policy changed')
    if config['npe']['batch_size'] % len(config['training_domains']):
        raise ValueError('Batch size cannot be balanced')


def optimize_density(density, train, validation, spec, seed, *, epoch_steps=None):
    """Pure torch loop. Input tuples are (theta, standardized_x, domain labels).

    All validation observations contribute; final partial batches are retained.
    Best state is copied, restored, and selected solely by domain-weighted NLL.
    """
    import torch
    theta, x, domain = train
    vt, vx, vd = validation
    labels = np.unique(domain)
    pools = [np.flatnonzero(domain == d) for d in labels]
    if not np.array_equal(labels, np.unique(vd)):
        raise ValueError('Training and validation domains disagree')
    if epoch_steps is None:
        epoch_steps = math.ceil(max(np.sum(domain == d) for d in labels)/spec['batch_size'])
    optimizer = torch.optim.Adam(density.parameters(), lr=spec['learning_rate'])
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, stale = np.inf, None, None, 0
    history = []
    for epoch in range(spec['max_num_epochs']):
        density.train()
        running = 0.
        for _ in range(epoch_steps):
            indices = draw_batch_from_pools(pools, spec['batch_size'], rng)
            idx = torch.as_tensor(indices, device=x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = -density.log_prob(theta[idx], condition=x[idx]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(density.parameters(), 5.)
            optimizer.step()
            running += float(loss.detach())
        density.eval()
        scores = {}
        with torch.no_grad():
            for d in labels:
                indices = np.flatnonzero(vd == d)
                total = 0.
                for start in range(0, len(indices), 1024):
                    idx = torch.as_tensor(indices[start:start+1024], device=vx.device)
                    total += float(-density.log_prob(vt[idx], condition=vx[idx]).sum())
                scores[str(d)] = total/len(indices)
        objective = float(np.mean(list(scores.values())))
        if not np.isfinite(objective):
            raise FloatingPointError('Non-finite validation objective')
        history.append(dict(epoch=epoch+1, updates=(epoch+1)*epoch_steps,
            train_nll=running/epoch_steps, validation_nll=objective, validation_by_domain=scores))
        if objective < best:
            best, best_state, best_epoch, stale = objective, deepcopy(density.state_dict()), epoch+1, 0
        else:
            stale += 1
        print(f'epoch={epoch+1} updates={(epoch+1)*epoch_steps} validation_nll={objective:.6f}', flush=True)
        if stale >= spec['stop_after_epochs']:
            break
    if best_state is None:
        raise RuntimeError('No finite validation checkpoint')
    density.load_state_dict(best_state)
    density.eval()
    return dict(history=history, best_epoch=best_epoch, best_validation_nll=best,
        epoch_steps=epoch_steps, stopped_early=stale >= spec['stop_after_epochs'],
        expected_rows_per_domain_per_epoch=epoch_steps*spec['batch_size']/len(labels),
        unique_train_counts={str(d):int(np.sum(domain == d)) for d in labels})


def train_controlled_npe(train, validation, config, device='cuda'):
    import torch
    from sbi.neural_nets import posterior_nn
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    validate_final_config(config)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    X, theta, domains = train
    VX, vtheta, vdomains = validation
    scaler = fit_training_scaler(X, domains)
    x = torch.as_tensor(scaler.transform(X), dtype=torch.float32, device=device)
    vx = torch.as_tensor(scaler.transform(VX), dtype=torch.float32, device=device)
    theta = torch.as_tensor(theta[:,None], dtype=torch.float32, device=device)
    vt = torch.as_tensor(vtheta[:,None], dtype=torch.float32, device=device)
    spec = config['npe']
    builder = posterior_nn(model=spec['model'], hidden_features=spec['hidden_features'],
        num_transforms=spec['num_transforms'], z_score_x='none')
    density = builder(theta, x).to(device)
    summary = optimize_density(density, (theta,x,domains), (vt,vx,vdomains), spec, config['seed'])
    # Empirical catalogue targets, not prior draws. BoxUniform only restricts support.
    prior = BoxUniform(low=torch.tensor([spec['prior_low']],device=device),
                       high=torch.tensor([spec['prior_high']],device=device))
    posterior = SNPE(prior=prior, device=device).build_posterior(density)
    return posterior, density, scaler, summary
