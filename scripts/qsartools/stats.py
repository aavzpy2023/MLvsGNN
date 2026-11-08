"""Statistics for comparing models evaluated by (repeated) cross-validation or on one test set.

Corrected resampled t-test (Nadeau & Bengio, 2003, Machine Learning 52:239-281, Table 1, statistic 6): the training sets
of resampled splits overlap, so the per-split scores are correlated and the naive standard error of their mean is too
small; the variance of the mean is taken as (1/J + n_test/n_train) s^2 instead of s^2/J, with a t distribution on J - 1
degrees of freedom. Nadeau & Bengio derive it for J independent random train/test splits; its use with the J = K x R
folds of repeated K-fold cross-validation is the corrected repeated k-fold cv test of Bouckaert & Frank (2004, PAKDD,
LNCS 3056:3-12).
"""
import numpy as np
from scipy import stats


def _corrected_se(values, n_train, n_test):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(values.var(ddof=1) * (1.0 / len(values) + n_test / n_train)))


def corrected_ttest(diff, n_train, n_test, popmean=0.0, margin=None):
    """Corrected resampled t-test of the mean of per-fold values `diff` against `popmean`.

    For a paired comparison of two models scored on the same folds, pass the per-fold differences (A minus B) and
    popmean 0. The equivalence test is the two one-sided tests procedure of Schuirmann (1987, J. Pharmacokinet.
    Biopharm. 15:657-680), here on the corrected standard error: equivalence at level alpha is declared exactly when the
    1 - 2 alpha interval lies inside (-margin, margin).

    Args:
        diff: per-fold values (one per fold of every repeat), e.g. score of model A minus score of model B.
        n_train: number of training samples of a fold (the mean, if folds differ in size).
        n_test: number of test samples of a fold (the mean, if folds differ in size).
        popmean: value the mean of `diff` is tested against.
        margin: optional equivalence margin (same units as `diff`); None skips the equivalence test.

    Returns:
        Dict with mean_diff (mean minus popmean), se (corrected standard error), t, df and the two-sided p. With
        `margin`, also an equivalence test (two one-sided tests, TOST) of |mean - popmean| < margin: p_equiv is the
        larger of the two one-sided p-values, and [ci90_low, ci90_high] is the 90% interval of the difference, which lies
        inside (-margin, margin) exactly when equivalence is declared at the 5% level.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean = float(diff.mean() - popmean)
    se = _corrected_se(diff, n_train, n_test)
    df = n - 1
    t = mean / se if se > 0 else 0.0
    out = {"mean_diff": mean, "se": se, "t": t, "df": df, "p": 2 * stats.t.sf(abs(t), df=df) if se > 0 else 1.0}
    if margin is not None:
        t_crit = stats.t.ppf(0.95, df=df)
        out["ci90_low"], out["ci90_high"] = mean - t_crit * se, mean + t_crit * se
        if se > 0:
            p_lower = stats.t.sf((mean + margin) / se, df=df)
            p_upper = stats.t.cdf((mean - margin) / se, df=df)
            out["p_equiv"] = float(max(p_lower, p_upper))
        else:
            out["p_equiv"] = 0.0 if abs(mean) < margin else 1.0
    return out


def holm(p_values):
    """Holm-adjusted p-values, returned in the input order: p_adj(i) = max over j <= i of min(1, (n - j + 1) p(j)) for
    the p-values sorted in increasing order (adjusted p-values: Wright, 1992, Biometrics 48:1005-1013). Rejecting every
    hypothesis with p_adj <= alpha is the sequentially rejective Bonferroni test of Holm (1979, Scand. J. Stat. 6:65-70).

    Args:
        p_values: raw p-values of the family of tests to be corrected together.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def r2(y_true, y_pred):
    """Coefficient of determination, 1 - SS_res / SS_tot (equal to sklearn.metrics.r2_score for one output, up to
    floating-point rounding).

    Args:
        y_true: observed values.
        y_pred: predicted values, in the same order.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def bootstrap_distribution(y_true, y_pred, metric=r2, n_boot=2000, seed=42):
    """Distribution of `metric(y_true, y_pred)` over n_boot resamples (with replacement) of the test molecules.

    The resampled indices depend only on `seed`, n_boot and the number of molecules, so two models scored with the same
    seed on the same molecules see the same resamples, and the difference of their distributions is a paired bootstrap.

    Args:
        y_true: observed values of the test molecules.
        y_pred: predicted values, in the same order.
        metric: callable metric(y_true, y_pred) -> float (r2 by default; e.g. sklearn.metrics.mean_absolute_error).
        n_boot: number of bootstrap resamples.
        seed: seed of the numpy RandomState that draws the resampled indices.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    out = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        out.append(metric(y_true[idx], y_pred[idx]))
    return np.array(out)


def percentile_interval(distribution, level=0.95):
    """(mean, lower, upper) of a bootstrap distribution, with the percentile interval at the given level.

    Args:
        distribution: bootstrap values (e.g. the output of bootstrap_distribution, or a difference of two of them).
        level: coverage of the interval (0.95 gives the 2.5th and 97.5th percentiles).
    """
    distribution = np.asarray(distribution, dtype=float)
    lo, hi = np.percentile(distribution, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(np.mean(distribution)), float(lo), float(hi)


def conformal_halfwidth(residuals, alpha=0.1):
    """Half-width of a split-conformal prediction interval (Vovk et al., 2005; Lei et al., 2018, JASA 113:1094-1111).

    The interval prediction +/- halfwidth covers the target of a new molecule with probability at least 1 - alpha,
    provided that molecule is exchangeable with the calibration set and that the model was fitted without it. The
    guarantee is finite-sample and distribution-free, and it is marginal: it holds on average over the population the
    calibration set is drawn from, not conditionally on a given molecule, so it says nothing about where in the
    chemical space the interval is wider. The quantile is therefore taken at rank ceil((n + 1) (1 - alpha)) of the n
    sorted absolute residuals rather than at the empirical (1 - alpha) quantile; when that rank exceeds n the bound is
    infinite, which is returned as inf.

    Args:
        residuals: residuals of the calibration set, y_true - y_pred; only their absolute value is used.
        alpha: miscoverage level in (0, 1); 0.1 gives 90% coverage.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), not {alpha}")
    absolute = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    if absolute.size == 0:
        raise ValueError("residuals must not be empty")
    rank = int(np.ceil((absolute.size + 1) * (1.0 - alpha)))
    return float("inf") if rank > absolute.size else float(absolute[rank - 1])


def bootstrap_p_value(distribution, reference):
    """Two-sided empirical p-value of `reference` against a bootstrap distribution: twice the smaller tail fraction,
    capped at 1.

    Args:
        distribution: bootstrap values.
        reference: value tested (e.g. a published score, or 0 for a distribution of differences).
    """
    distribution = np.asarray(distribution, dtype=float)
    p = 2.0 * min(np.mean(distribution <= reference), np.mean(distribution >= reference))
    return float(min(1.0, p))
