from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit


def pair_groups(frame: pd.DataFrame) -> np.ndarray:
    hosts = np.unique(np.concatenate([frame["host_i"].to_numpy(), frame["host_j"].to_numpy()]))
    lookup = {host: idx for idx, host in enumerate(hosts)}
    rows = frame["host_i"].map(lookup).to_numpy()
    cols = frame["host_j"].map(lookup).to_numpy()
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(hosts), len(hosts)))
    _, labels = connected_components(graph, directed=False)
    return labels[rows]


def group_split(frame: pd.DataFrame, test_size: float = 0.15, seed: int = 42):
    groups = pair_groups(frame)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(frame, groups=groups))
    if set(groups[train]) & set(groups[test]):
        raise RuntimeError("Connected host components overlap across train/test split")
    return train, test, groups
