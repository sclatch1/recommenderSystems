import numpy as np
import scipy.sparse as sp
from recpack.matrix import InteractionMatrix

from src.splitter import PPACEqualExposureSplitter


def _toy_interaction_matrix(n_users=60, n_items=20, density=0.3, seed=0):
    mat = sp.random(n_users, n_items, density=density, random_state=seed, data_rvs=lambda n: np.ones(n)).tocsr()
    mat.data[:] = 1
    return InteractionMatrix.from_csr_matrix(mat)


def test_ppac_equal_exposure_splitter_enforces_flat_quota():
    """Every item that survives into val/test must have exactly the quota
    derived from exposure_frac - that flat, zero-variance quota is the
    whole point of this splitter (see PPACEqualExposureSplitter's docstring
    for how this was reverse-engineered from the paper's released data)."""
    data = _toy_interaction_matrix(n_users=60)
    # exposure_frac chosen so the derived quota (exposure_frac * n_users *
    # test_frac) is an exact, easy-to-check integer: 1/6 * 60 * 0.3 = 3.
    exposure_frac = 1 / 6
    expected_quota = 3

    splitter = PPACEqualExposureSplitter(test_frac=0.3, validation_frac=0.3, exposure_frac=exposure_frac, seed=42)
    train, val, test = splitter.split(data)

    for split in (val, test):
        counts = np.array(split.values.sum(axis=0)).ravel()
        nonzero_counts = counts[counts > 0]
        assert len(nonzero_counts) > 0, "expected at least one item to survive the exposure quota"
        assert np.all(nonzero_counts == expected_quota)

    # items/excess dropped during the stage-2 resample are never smuggled
    # back into train
    assert train.values.nnz + val.values.nnz + test.values.nnz <= data.values.nnz
