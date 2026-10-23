"""Gaussian-noise augmentation of a feature matrix, with the group labels needed to keep the copies of one sample
together when the augmented data are split again (e.g. by an inner cross-validation)."""
import numpy as np


def gaussian_noise_augmentation(X, y, noise_levels=(0.01, 0.001), seed=42, include_original=True):
    """One noisy copy of every row per noise level: X + N(0, sigma^2) element-wise, with the target unchanged.

    Rows are ordered [original block (if include_original)], then one block per noise level, each block in input order.
    The noise is drawn with numpy.random.RandomState(seed), one block after the other.

    Returns (X_aug, y_aug, groups): groups[i] is the index of the input row that output row i comes from, so that a
    GroupKFold on it never puts a copy of a sample on the other side of a split from that sample.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    X_parts, y_parts = ([X], [y]) if include_original else ([], [])
    for sigma in noise_levels:
        X_parts.append(X + rng.normal(0, sigma, X.shape))
        y_parts.append(y)
    n_blocks = len(X_parts)
    return np.vstack(X_parts), np.concatenate(y_parts), np.tile(np.arange(len(y)), n_blocks)
