from __future__ import annotations

import numpy as np


def train_npe(X_train: np.ndarray, theta_train: np.ndarray, config: dict, device: str):
    import torch
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = config["npe"]
    density_builder = posterior_nn(
        model=spec["model"],
        hidden_features=int(spec["hidden_features"]),
        num_transforms=int(spec["num_transforms"]),
    )
    prior = BoxUniform(
        low=torch.tensor([float(spec["prior_low"])], device=device),
        high=torch.tensor([float(spec["prior_high"])], device=device),
    )
    inference = SNPE(prior=prior, density_estimator=density_builder, device=device)
    inference.append_simulations(
        torch.tensor(theta_train.reshape(-1, 1), dtype=torch.float32, device=device),
        torch.tensor(X_train, dtype=torch.float32, device=device),
    )
    density = inference.train(
        training_batch_size=int(spec["batch_size"]),
        learning_rate=float(spec["learning_rate"]),
        max_num_epochs=int(spec["max_num_epochs"]),
        stop_after_epochs=int(spec["stop_after_epochs"]),
        show_train_summary=False,
    )
    return inference.build_posterior(density), inference, density


def posterior_draws(posterior, X: np.ndarray, draws: int, batch_size: int, device: str) -> np.ndarray:
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            values = posterior.sample_batched((draws,), x=batch).squeeze(-1)
            chunks.append(values.cpu().numpy())
    return np.concatenate(chunks, axis=1)
