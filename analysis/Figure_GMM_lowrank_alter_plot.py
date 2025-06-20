
"""
This script is used to visualize the approximation error of EDM score by GMM score. 
As a function of the rank of the GMM and the number of modes, focusing on the MNIST and CIFAR10 datasets.
It assumes the computation has been done and the results are stored in the csv files in the `Tables` directory.
It plot it in a different way than the previous script, focusing on the contribution of 
the number of modes beyond the Gaussian score, and the minimal required rank to achieve a certain level of approximation.
"""
#%%
# %load_ext autoreload
# %autoreload 2
#%%
import os
import sys
from os.path import join
from sympy import plot
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import seaborn as sns
sys.path.append("../")
from gaussian_teleport.utils.plot_utils import saveallforms
# set pandas display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#%%
def extract_res_mats(df_gmm_rk, varname="St_residual", 
                     n_clusters_list=None, n_rank_list=None, 
                     sigmas=None, ):
    
    # sigmas = [1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ]
    # n_clusters_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    # n_rank_list = [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]
    if sigmas is None:
        sigmas = sorted(df_gmm_rk.sigma.unique())
    if n_clusters_list is None:
        n_clusters_list = sorted(df_gmm_rk.n_cluster.unique())
        # remove the nan value
        n_clusters_list = [int(x) for x in n_clusters_list if not np.isnan(x)]
    if n_rank_list is None:
        n_rank_list = sorted(df_gmm_rk.n_rank.unique())
        n_rank_list = [int(x) for x in n_rank_list if not np.isnan(x)]
    res_mats = {}
    res_mat_pivots = {}
    for sigma in sigmas:
        res_mat = []
        for n_clusters in n_clusters_list:
            for n_rank in n_rank_list:
                res = df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)][varname].values
                # Ensure the result is not empty before appending
                if res.size > 0:
                    res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, varname: res[0]})
                else:
                    # Handle the case where the result might be empty
                    res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, varname: None})
        
        res_mat = pd.DataFrame(res_mat)
        res_mats[sigma] = res_mat
        res_mat_pivot = res_mat.pivot_table(index="n_clusters", columns="n_rank",
                                           values=varname, aggfunc="mean")
        res_mat_pivots[sigma] = res_mat_pivot
    return res_mats, res_mat_pivots


def lineplot_with_log_color_scale(data, x_col, y_col, hue_col, cmap='turbo', 
                                figsize=(6, 4), ax=None, legend=True, colorbar=False, 
                                **kwargs):
    """
    Plots data with colors mapped to a hue column on a logarithmic scale.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x_col: String, name of the column to use for the x-axis.
    - y_col: String, name of the column to use for the y-axis.
    - hue_col: String, name of the column to map colors to on a logarithmic scale.
    - cmap: String, name of the matplotlib colormap to use.
    - figsize: Tuple of int, size of the figure.
    - lw: Float, linewidth of the plot lines.
    - marker: String, marker style.
    - markersize: Int, size of the markers.
    - alpha: Float, opacity level of the plot lines.
    - linestyle: String, style of the plot lines.
    """
    # Set figure size
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    # Create a logarithmic normalization instance for hue_col
    log_norm = mcolors.LogNorm(vmin=data[hue_col].min(), vmax=data[hue_col].max())
    # Create a colormap
    cmap = plt.get_cmap(cmap)
    # Plot each set of points with a unique hue_col value
    for hue_value in data[hue_col].unique():
        subset = data[data[hue_col] == hue_value]
        ax.plot(subset[x_col], subset[y_col], 
                 color=cmap(log_norm(hue_value)), **kwargs, 
                 label=hue_value)
    # Set log scale for x and y axes
    ax.set_yscale('log')
    ax.set_xscale('log')
    # Adjust y limits
    ylim = ax.get_ylim()
    ax.set_ylim([None, min(1, ylim[1])])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if legend:
        ax.legend(title=hue_col)
    if colorbar:
        # Create colorbar as legend for n_rank
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=log_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('n_rank')
#%%

def visualize_gmm_lowrank_residual(res_mats, xvar="n_rank", yvar="residual", huevar="n_clusters", 
                       sigmas=[1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 
                               1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ], 
                       figsize=(22.5, 8), nrowcols=(2, 5), runname="MNIST miniEDM",
                       savename="MNIST_GMM_lowrank_residual_nrank_hue_Ncomp",):
    """
    Plots the score residual of GMM MNIST dataset for varying sigma values.

    Parameters:
    - df_gmm_rk: DataFrame containing the GMM data.
    - sigmas: List of sigma values to plot.
    - n_clusters_list: List of n_clusters values to consider.
    - n_rank_list: List of n_rank values to consider.
    - figsize: Tuple indicating figure size.
    """
    if xvar == "n_rank":
        xname = "Rank of Mode"
    elif xvar == "n_clusters":
        xname = "Num of Modes"
    if huevar == "n_rank":
        hname = "Rank of Mode"
    elif huevar == "n_clusters":
        hname = "Num of Modes"
    if yvar == "St_residual":
        varname = "Score"
    elif yvar == "Dt_residual":
        varname = "Denoiser"
    nrow, ncol = nrowcols
    figh, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.flatten()
    for i, sigma in enumerate(sigmas):
        res_mat = res_mats[sigma]
        # Assuming `lineplot_with_log_color_scale` is a previously defined function
        lineplot_with_log_color_scale(res_mat, xvar, yvar, huevar, 
                                      cmap="turbo", ax=axs[i], legend=False,
                                      lw=1.5, marker="o", markersize=5, alpha=0.4)
        # increase font size of minor ticks
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=16)
        axs[i].set_title(f"sigma={sigma}", fontsize=20, y=0.94)
        axs[i].set_ylabel(f"{varname} EV Residual", fontsize=18)
        axs[i].set_xlabel(xname, fontsize=18)
        # set y-axis label for the left column
        if not (i % ncol == 0):
            axs[i].set_ylabel("")
        # Set x-axis label for the bottom row others as empty
        if not (i >= (ncol * (nrow - 1))):
            axs[i].set_xlabel("")
        if i == (ncol * nrow - 1):
            axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18,
                          title=hname, title_fontsize=18)
    
    plt.suptitle(f"{varname} Residual of EDM vs GMM score | {runname} ", fontsize=24)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # plt.tight_layout()
    saveallforms(figsumdir, savename, figh, ["pdf", "png"])
    plt.show()
    return figh, axs


def visualize_gmm_lowrank_residual_heatmap(res_mats, 
                        yvar="n_rank", xvar="n_clusters", plotvar="St_residual",
                       sigmas=[1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 
                               1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ], 
                       figsize=(22.5, 8), nrowcols=(2, 5), runname="MNIST miniEDM",
                       savename="MNIST_GMM_lowrank_residual_heatmap",):
    """
    Plots the score residual of GMM MNIST dataset for varying sigma values.

    Parameters:
    - df_gmm_rk: DataFrame containing the GMM data.
    - sigmas: List of sigma values to plot.
    - n_clusters_list: List of n_clusters values to consider.
    - n_rank_list: List of n_rank values to consider.
    - figsize: Tuple indicating figure size.
    """
    if xvar == "n_rank":
        xname = "Rank of Mode"
    elif xvar == "n_clusters":
        xname = "Num of Modes"
    if plotvar == "St_residual":
        varname = "Score"
    elif plotvar == "Dt_residual":
        varname = "Denoiser"
    nrow, ncol = nrowcols
    figh, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.flatten()
    for i, sigma in enumerate(sigmas):
        res_mat = res_mats[sigma]
        res_mat_pivot = res_mat.pivot_table(index=xvar, columns=yvar,
                            values=plotvar, aggfunc="mean")
        sns.heatmap(res_mat_pivot, annot=True, fmt=".2f", 
            cmap="YlGnBu", ax=axs[i])#norm=norm_scale)
        axs[i].set_title(f"sigma={sigma}", fontsize=16)
        # axs[i].set_ylabel("Score EV Residual", fontsize=14)
        # axs[i].set_xlabel(xname, fontsize=14)
        # set y-axis label for the left column
        if not (i % ncol == 0):
            axs[i].set_ylabel("")
        # Set x-axis label for the bottom row others as empty
        if not (i >= (ncol * (nrow - 1))):
            axs[i].set_xlabel("")
    
    plt.suptitle(f"{varname} Residual of GMM {runname} Dataset ", fontsize=20)
    plt.tight_layout()
    saveallforms(figsumdir, savename, figh, ["pdf", "png"])
    plt.show()
    return figh, axs


def visualize_gmm_lowrank_residual_heatmap_separate(res_mats, 
                        yvar="n_clusters", xvar="n_rank", plotvar="St_residual",
                       sigmas=[1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 
                               1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ], 
                       figsize=(7, 6), runname="MNIST miniEDM",
                       savename="MNIST_GMM_lowrank_residual_heatmap",):
    """
    Plots the score residual of GMM MNIST dataset for varying sigma values.

    Parameters:
    - df_gmm_rk: DataFrame containing the GMM data.
    - sigmas: List of sigma values to plot.
    - n_clusters_list: List of n_clusters values to consider.
    - n_rank_list: List of n_rank values to consider.
    - figsize: Tuple indicating figure size.
    """
    if xvar == "n_rank":
        xname = "Rank of Mode"
    elif xvar == "n_clusters":
        xname = "Num of Modes"
    if yvar == "n_rank":
        yname = "Rank of Mode"
    elif yvar == "n_clusters":
        yname = "Num of Modes"
    if plotvar == "St_residual":
        varname = "Score"
    elif plotvar == "Dt_residual":
        varname = "Denoiser"
    for i, sigma in enumerate(sigmas):
        res_mat = res_mats[sigma]
        res_mat_pivot = res_mat.pivot_table(index=yvar, columns=xvar,
                            values=plotvar, aggfunc="mean")
        # use this to show only the significant digits in the heatmap, not the exponent
        magnif_const = 10**(np.round(np.log10(res_mat_pivot.min().min())))
        # def custom_format(x):
        #     return f"{x:.1e}".split('e')[0]
        def custom_format(x):
            return f"{x / magnif_const:.1f}"
        res_pivot_fmt = res_mat_pivot.applymap(custom_format)
        figh, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(res_mat_pivot, annot=res_pivot_fmt, fmt='',
            cmap="YlGnBu", ax=ax) 
        # sns.heatmap(res_mat_pivot, annot=True, fmt=".2f", 
        #     cmap="YlGnBu", ax=ax)#norm=norm_scale)
        ax.set_title(f"{varname} Residual of EDM vs GMM (x {magnif_const:.0e})\n {runname} | sigma={sigma}", fontsize=17)
        ax.set_ylabel(yname, fontsize=16)
        ax.set_xlabel(xname, fontsize=16)
        plt.tight_layout()
        saveallforms(figsumdir, savename+f"_sigma{sigma}", figh, ["pdf", "png"])
        plt.show()
    return

#%%
rootdir = "../" # e.g. "/Users/binxuwang/Github/GaussianTeleportationDiffusion"
tabdir = join(rootdir, "Tables")
figroot = join(rootdir, "Figures") # can be changed to any other directory
figsumdir = join(figroot, "GMM_lowrk_approx_summary")
os.makedirs(figsumdir, exist_ok=True)

# %% [markdown]
# ### MNIST score with varying rank and components
df_gmm_rk = pd.read_csv(join(tabdir, "MNIST_edm_1000k_epoch_gmm_exp_var_gmm_rk.csv"))
model_name = "MNIST_miniEDM"
# preprocess the dataframe to extact the rank and components
df_gmm_rk["St_residual"] = 1 - df_gmm_rk["St_EV"]
df_gmm_rk["Dt_residual"] = 1 - df_gmm_rk["Dt_EV"]
df_gmm_rk[['n_cluster', 'n_rank']] = df_gmm_rk['name'].str.extract(r'gmm_(\d+)_mode_(\d+)_rank')
df_gmm_rk['n_cluster'] = df_gmm_rk['n_cluster'].astype('Int64')
df_gmm_rk['n_rank'] = df_gmm_rk['n_rank'].astype('Int64')

#%% # effectiveness of adding more Gaussian modes
# create a new column, improved residual
df_gmm_rk["St_residual_improv"] = df_gmm_rk["St_residual"]
for sigma in df_gmm_rk.sigma.unique():
    for n_rank in df_gmm_rk.n_rank.unique():
        rank_sigma_mask = (df_gmm_rk.n_rank == n_rank) & (df_gmm_rk.sigma == sigma)
        print ("Rank", n_rank)
        print (df_gmm_rk[rank_sigma_mask
                        & (df_gmm_rk.n_cluster == 1)]["St_residual"].max())
        print (df_gmm_rk[rank_sigma_mask].shape)
        print (df_gmm_rk[rank_sigma_mask & (df_gmm_rk.n_cluster == 1)].shape)
        df_gmm_rk.loc[rank_sigma_mask,  "St_residual_improv"] = \
            df_gmm_rk.loc[rank_sigma_mask & (df_gmm_rk.n_cluster == 1), "St_residual"].max() -\
            df_gmm_rk.loc[rank_sigma_mask,"St_residual"]

#%%
figh, ax = plt.subplots(figsize=(5, 5))
lineplot_with_log_color_scale(df_gmm_rk[(df_gmm_rk.n_rank == 1024) &
            df_gmm_rk.n_cluster.isin([2, 10, 100, 200, 1000])],
            "sigma", "St_residual_improv", "n_cluster", 
            cmap="plasma", ax=ax, marker="o", markersize=5, alpha=0.8)
plt.xscale("log")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(title="n_cluster", fontsize=12, title_fontsize=14)
# xlabel ylabel fontsize
ax.set_xlabel("sigma", fontsize=16)
ax.set_ylabel("∆ fraction of Exp Var", fontsize=16)
ax.set_title(f"Contribution of # of Mode beyond Gaussian\n{model_name}", fontsize=16)
saveallforms(figsumdir, f"{model_name}_GMM_ncomp_contrib_across_sigma", figh, ["pdf", "png"])

#%% Minimal required rank

thresh_fraction = 1.1
full_rank = df_gmm_rk.n_rank.max()
# compute the minimal required rank 
df_gmm_rk["St_residual_degrad_ratio"] = 0
for sigma in df_gmm_rk.sigma.unique():
    for n_cluster in df_gmm_rk.n_cluster.unique():
        cluster_sigma_mask = (df_gmm_rk.n_cluster == n_cluster) & (df_gmm_rk.sigma == sigma)
        print ("n_cluster", n_cluster)
        print (df_gmm_rk[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank)]["St_residual"].max())
        print (df_gmm_rk[cluster_sigma_mask].shape)
        print (df_gmm_rk[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank)].shape)
        df_gmm_rk.loc[cluster_sigma_mask,"St_residual_degrad_ratio"] = \
            df_gmm_rk.loc[cluster_sigma_mask,"St_residual"] / \
            df_gmm_rk.loc[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank), "St_residual"].max()

min_rank_df = []
for sigma in df_gmm_rk.sigma.unique():
    for n_cluster in df_gmm_rk.n_cluster.unique():
        cluster_sigma_mask = (df_gmm_rk.n_cluster == n_cluster) & (df_gmm_rk.sigma == sigma)
        min_rank = df_gmm_rk.loc[cluster_sigma_mask & (df_gmm_rk.St_residual_degrad_ratio < thresh_fraction), "n_rank"].min()
        print (f"Minimal rank for sigma={sigma}, n_cluster={n_cluster} is {min_rank}")
        min_rank_df.append({"sigma": sigma, "n_cluster": n_cluster, "min_rank": min_rank})
min_rank_df = pd.DataFrame(min_rank_df)
#%%
figh, ax = plt.subplots(figsize=(5, 5))
lineplot_with_log_color_scale(min_rank_df[
    min_rank_df.n_cluster.isin([1, 10, 100, 1000])], 
                x_col="sigma", y_col="min_rank", hue_col="n_cluster", 
                ax=ax, marker="o", cmap="viridis",alpha=0.8)
ax.set_ylim(5, full_rank)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(title="n_cluster", fontsize=12, title_fontsize=14)
# xlabel ylabel fontsize
ax.set_xlabel("sigma", fontsize=16)
ax.set_ylabel("Minimal Rank", fontsize=16)
ax.set_title(f"Minimal Rank (increase residual EV < {thresh_fraction})\n{model_name}", fontsize=16)
saveallforms(figsumdir, f"{model_name}_GMM_min_rank_{thresh_fraction}_across_sigma", figh, ["pdf", "png"])
figh.show()
#%%
#%% [markdown]
# ### CIFAR10 score with varying rank and components
df_gmm_rk = pd.read_csv(join(tabdir, "cifar10_uncond_edm_vp_pretrained_epoch_gmm_exp_var_gmm_rk.csv"))
model_name = "CIFAR10_uncond_EDM"
# preprocess the dataframe to extact the rank and components
df_gmm_rk["St_residual"] = 1 - df_gmm_rk["St_EV"]
df_gmm_rk["Dt_residual"] = 1 - df_gmm_rk["Dt_EV"]
df_gmm_rk[['n_cluster', 'n_rank']] = df_gmm_rk['name'].str.extract(r'gmm_(\d+)_mode_(\d+)_rank')
df_gmm_rk['n_cluster'] = df_gmm_rk['n_cluster'].astype("Int64")
df_gmm_rk['n_rank'] = df_gmm_rk['n_rank'].astype("Int64")
#%%
#%% # effectiveness of adding more Gaussian modes
# create a new column, improved residual
df_gmm_rk["St_residual_improv"] = df_gmm_rk["St_residual"]
for sigma in df_gmm_rk.sigma.unique():
    for n_rank in df_gmm_rk.n_rank.unique():
        rank_sigma_mask = (df_gmm_rk.n_rank == n_rank) & (df_gmm_rk.sigma == sigma)
        print ("Rank", n_rank)
        print (df_gmm_rk[rank_sigma_mask
                        & (df_gmm_rk.n_cluster == 1)]["St_residual"].max())
        # print (df_gmm_rk[rank_sigma_mask].shape)
        # print (df_gmm_rk[rank_sigma_mask & (df_gmm_rk.n_cluster == 1)].shape)
        assert len(df_gmm_rk[rank_sigma_mask & (df_gmm_rk.n_cluster == 1)]) == 1 or pd.isna(n_rank)
        df_gmm_rk.loc[rank_sigma_mask,  "St_residual_improv"] = \
            df_gmm_rk.loc[rank_sigma_mask & (df_gmm_rk.n_cluster == 1), "St_residual"].max() - \
            df_gmm_rk.loc[rank_sigma_mask,"St_residual"]

#%%
full_rank = df_gmm_rk.n_rank.max()
figh, ax = plt.subplots(figsize=(5, 5))
lineplot_with_log_color_scale(df_gmm_rk[(df_gmm_rk.n_rank == full_rank) &
            df_gmm_rk.n_cluster.isin([2, 10, 50, 100, ])],
            "sigma", "St_residual_improv", "n_cluster", 
            cmap="plasma", ax=ax, marker="o", markersize=5, alpha=0.8)
plt.xscale("log")
# plt.yscale("linear")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(title="n_cluster", fontsize=12, title_fontsize=14)
# xlabel ylabel fontsize
ax.set_xlabel("sigma", fontsize=16)
ax.set_ylabel("∆ fraction of Exp Var", fontsize=16)
ax.set_title(f"Contribution of # of Mode beyond Gaussian\n{model_name}", fontsize=16)
saveallforms(figsumdir, f"{model_name}_GMM_ncomp_contrib_across_sigma", figh, ["pdf", "png"])

#%% Minimal required rank
thresh_fraction = 1.1
full_rank = df_gmm_rk.n_rank.max()
# compute the minimal required rank 
df_gmm_rk["St_residual_degrad_ratio"] = 0
for sigma in df_gmm_rk.sigma.unique():
    for n_cluster in df_gmm_rk.n_cluster.unique():
        cluster_sigma_mask = (df_gmm_rk.n_cluster == n_cluster) & (df_gmm_rk.sigma == sigma)
        print ("n_cluster", n_cluster)
        print (df_gmm_rk[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank)]["St_residual"].max())
        # print (df_gmm_rk[cluster_sigma_mask].shape)
        # print (df_gmm_rk[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank)].shape)
        assert len(df_gmm_rk[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank)]) == 1 or pd.isna(n_cluster)
        df_gmm_rk.loc[cluster_sigma_mask,"St_residual_degrad_ratio"] = \
            df_gmm_rk.loc[cluster_sigma_mask,"St_residual"] / \
            df_gmm_rk.loc[cluster_sigma_mask & (df_gmm_rk.n_rank == full_rank), "St_residual"].max()

min_rank_df = []
for sigma in df_gmm_rk.sigma.unique():
    for n_cluster in df_gmm_rk.n_cluster.unique():
        cluster_sigma_mask = (df_gmm_rk.n_cluster == n_cluster) & (df_gmm_rk.sigma == sigma)
        min_rank = df_gmm_rk.loc[cluster_sigma_mask & (df_gmm_rk.St_residual_degrad_ratio < thresh_fraction), "n_rank"].min()
        # print (f"Minimal rank for sigma={sigma}, n_cluster={n_cluster} is {min_rank}")
        min_rank_df.append({"sigma": sigma, "n_cluster": n_cluster, "min_rank": min_rank})
min_rank_df = pd.DataFrame(min_rank_df)
#%%
figh, ax = plt.subplots(figsize=(5, 5))
lineplot_with_log_color_scale(min_rank_df[
    min_rank_df.n_cluster.isin([1, 10, 100, 200])], 
                x_col="sigma", y_col="min_rank", hue_col="n_cluster", 
                ax=ax, marker="o", cmap="viridis",alpha=0.8)
ax.set_ylim(5, full_rank+500)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(title="n_cluster", fontsize=12, title_fontsize=14)
# xlabel ylabel fontsize
ax.set_xlabel("sigma", fontsize=16)
ax.set_ylabel("Minimal Rank", fontsize=16)
ax.set_title(f"Minimal Rank (increase residual EV < {thresh_fraction})\n{model_name}", fontsize=16)
saveallforms(figsumdir, f"{model_name}_GMM_min_rank_{thresh_fraction}_across_sigma", figh, ["pdf", "png"])
figh.show()

