import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_ind, ttest_1samp


def ttest_rel_df(df, msk, col1, col2):
    return ttest_rel(df[msk][col1], df[msk][col2])


def ttest_ind_df(df, msk1, msk2, col):
    return ttest_ind(df[msk1][col], df[msk2][col])


def ttest_1samp_print(seq, scalar):
    tval, pval = ttest_1samp(seq, scalar)
    print(f"{seq.mean():.3f}+-{seq.std():.3f} ~ {scalar:.3f} tval: {tval:.2f}, pval: {pval:.1e}")
    return tval, pval


def latexify_tststs(tval, pval, df):
    latex_stat_str = "t_{%d}=%.3f " % (df, tval) + ("p=%.1e}" % pval).replace("e", r"\times 10^{").replace("+0", "").replace("-0", "-")
    return latex_stat_str

def ttest_rel_print(seq1, seq2, sem=False, latex=False, nan_policy="omit"):
    tval, pval = ttest_rel(seq1, seq2, nan_policy=nan_policy)
    df = len(seq1) - 1
    latex_stat_str = latexify_tststs(tval, pval, df)
    if sem:
        sem1 = seq1.std() / np.sqrt(len(seq1))
        sem2 = seq2.std() / np.sqrt(len(seq2))
        result_str = f"{seq1.mean():.3f}+-{sem1:.3f} ~ {seq2.mean():.3f}+-{sem2:.3f} (N={len(seq1)}) t={tval:.2f}, p={pval:.1e} df={df}"
        latex_str = f"{seq1.mean():.3f}\pm{sem1:.3f} ~ {seq2.mean():.3f}\pm{sem2:.3f} (N={len(seq1)}) " + latex_stat_str
    else:
        result_str = f"{seq1.mean():.3f}+-{seq1.std():.3f} ~ {seq2.mean():.3f}+-{seq2.std():.3f} (N={len(seq1)}) t={tval:.2f}, p={pval:.1e} df={df}"
        latex_str = f"{seq1.mean():.3f}\pm{seq1.std():.3f} ~ {seq2.mean():.3f}\pm{seq2.std():.3f} (N={len(seq1)}) " + latex_stat_str
    if latex:
        print(latex_str)
    else:
        print(result_str)
    return tval, pval, result_str


def ttest_ind_print(seq1, seq2, sem=False, latex=False, nan_policy="omit"):
    tval, pval = ttest_ind(seq1, seq2, nan_policy=nan_policy)
    df = len(seq1) + len(seq2) - 2
    latex_stat_str = latexify_tststs(tval, pval, df)
    if sem:
        sem1 = seq1.std() / np.sqrt(len(seq1))
        sem2 = seq2.std() / np.sqrt(len(seq2))
        result_str = f"{seq1.mean():.3f}+-{sem1:.3f} (N={len(seq1)}) ~ {seq2.mean():.3f}+-{sem2:.3f} (N={len(seq2)}) t={tval:.2f}, p={pval:.1e} df={df}"
        latex_str = f"{seq1.mean():.3f}\pm{sem1:.3f} (N={len(seq1)}) ~ {seq2.mean():.3f}\pm{sem2:.3f} (N={len(seq2)}) " + latex_stat_str
    else:
        result_str = f"{seq1.mean():.3f}+-{seq1.std():.3f} (N={len(seq1)}) ~ {seq2.mean():.3f}+-{seq2.std():.3f} (N={len(seq2)}) t={tval:.2f}, p={pval:.1e} df={df}"
        latex_str = f"{seq1.mean():.3f}\pm{seq1.std():.3f} (N={len(seq1)}) ~ {seq2.mean():.3f}\pm{seq2.std():.3f} (N={len(seq2)}) " + latex_stat_str
    if latex:
        print(latex_str)
    else:
        print(result_str)
    return tval, pval, result_str


def ttest_rel_print_df(df, msk, col1, col2, sem=False, latex=False):
    if msk is None:
        msk = np.ones(len(df), dtype=bool)
    print(f"{col1} ~ {col2} (N={msk.sum()})", end=" ")
    return ttest_rel_print(df[msk][col1], df[msk][col2], sem=sem, latex=latex)


def ttest_ind_print_df(df, msk1, msk2, col, sem=False, latex=False):
    print(f"{col} (N={msk1.sum()}) ~ (N={msk2.sum()})", end=" ")
    return ttest_ind_print(df[msk1][col], df[msk2][col], sem=sem, latex=latex)


def paired_strip_plot(df, msk, col1, col2):
    if msk is None:
        msk = np.ones(len(df), dtype=bool)
    vec1 = df[msk][col1]
    vec2 = df[msk][col2]
    xjitter = 0.1 * np.random.randn(len(vec1))
    figh = plt.figure(figsize=[5, 6])
    plt.scatter(xjitter, vec1)
    plt.scatter(xjitter+1, vec2)
    plt.plot(np.arange(2)[:,None]+xjitter[None,:],
             np.stack((vec1, vec2)), color="k", alpha=0.1)
    plt.xticks([0,1], [col1, col2])
    tval, pval = ttest_rel_df(df, msk, col1, col2)
    plt.title(f"tval={tval:.3f}, pval={pval:.1e} N={msk.sum()}")
    plt.show()
    return figh


def shaded_errorbar(x, y, yerr, color, alpha=0.3, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(x, y, color=color, **kwargs)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)


def shaded_errorbar_arr(arr, color="r", alpha=0.3, errtype="sem", ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = np.arange(arr.shape[1])
    y = np.nanmean(arr, axis=0)
    if errtype == "sem":
        yerr = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
    elif errtype == "std":
        yerr = np.nanstd(arr, axis=0)
    else:
        raise Exception(f"errtype {errtype} not recognized")
    ax.plot(x, y, color=color, **kwargs)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha,
                     label="" if "label" not in kwargs else kwargs["label"]+"_"+errtype)


def scatter_corr(df, x, y, ax=None, corrtype="pearson", **kwargs):
    if ax is None:
        ax = plt.gca()
    # ax.scatter(df[x], df[y], **kwargs)
    sns.scatterplot(data=df, x=x, y=y, ax=ax, **kwargs)
    # scipy pearsonr
    validmsk = np.logical_and(np.isfinite(df[x]), np.isfinite(df[y]))
    if corrtype.lower() == "pearson":
        rho, pval = pearsonr(df[x][validmsk], df[y][validmsk], )
    elif corrtype.lower() == "spearman":
        rho, pval = spearmanr(df[x][validmsk], df[y][validmsk], )
    else:
        raise NotImplementedError
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs. {y}\ncorr={rho:.3f} p={pval:.1e} n={validmsk.sum()}")
    return ax, rho, pval


def trivariate_corr(x, y, z):
    """
    x = np.random.normal(0, 1, size=100)
    y = np.random.normal(0, 1, size=100)
    z = np.random.normal(0, 1, size=100)

    r = trivariate_corr(x, y, z)
    print(r)
    :param x:
    :param y:
    :param z:
    :return:
    """
    xy_corr = np.corrcoef(x, y)[0, 1]
    xz_corr = np.corrcoef(x, z)[0, 1]
    yz_corr = np.corrcoef(y, z)[0, 1]
    r = (xy_corr**2 + xz_corr**2 + yz_corr**2 + 2*xy_corr*xz_corr*yz_corr) / (1 - xy_corr**2 - xz_corr**2- yz_corr**2 + 2*xy_corr*xz_corr*yz_corr)
    return r


def test_correlation_diff(A, B, C, verbose=False):
    # Compute the correlations
    r_AB = np.corrcoef(A, B)[0, 1]
    r_AC = np.corrcoef(A, C)[0, 1]
    # Compute the standard error of the difference between the two z scores
    N = len(A)
    SE = np.sqrt((2 * (N - 3))**-1)
    # Apply Fisher's Z transformation
    z_AB = 0.5 * np.log((1 + r_AB) / (1 - r_AB)) / SE
    z_AC = 0.5 * np.log((1 + r_AC) / (1 - r_AC)) / SE
    # Compute the test statistic
    z = (z_AB - z_AC)
    # Compute the p-value
    p_value = 1 - stats.norm.cdf(z)
    p_value_rev = stats.norm.cdf(z)
    p_2sided = 2 * np.min([p_value, p_value_rev])
    if verbose:
        print(f"r_AB={r_AB:.3f}, r_AC={r_AC:.3f}, z_AB={z_AB:.3f}, z_AC={z_AC:.3f}"
              f"\nz={z:.3f}, p={p_2sided:.1e} (2 side), p={min(p_value, p_value_rev):.1e} (1 side) N={N:d}")
    return z, p_2sided


def test_correlation_diff_df(df, colA, colB, colC, verbose=False):
    return test_correlation_diff(df[colA], df[colB], df[colC], verbose=verbose)