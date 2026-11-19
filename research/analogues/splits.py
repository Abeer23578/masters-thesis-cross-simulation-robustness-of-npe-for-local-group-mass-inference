import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit

def pair_groups(df):
    """We consider nodes as halos and pairs as edges. Because of this no galaxy can land on both sides of a split"""
    hosts = np.unique(np.concatenate([df["host_i"].values, df["host_j"].values]))
    idx = {h: k for k, h in enumerate(hosts)}
    r = df["host_i"].map(idx).values
    c = df["host_j"].map(idx).values
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(hosts), len(hosts)))
    _, labels = connected_components(g, directed=False)
    return labels[r]

def group_split(df, test_size=0.15, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=pair_groups(df)))
