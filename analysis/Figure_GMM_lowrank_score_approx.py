
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
import seaborn as sns
from torchvision.utils import make_grid, save_image
# from core.gmm_special_dynamics import alpha
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionMemorization")
# from train_edm import edm_sampler, EDM, create_model
# from core.edm_utils import get_default_config, create_edm
from gaussian_teleport.utils.plot_utils import saveallforms
# set pandas display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

figroot = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/DiffusionHiddenLinear"
figsumdir = join(figroot, "GMM_lowrk_approx_summary")
os.makedirs(figsumdir, exist_ok=True)

#%%
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

# %% [markdown]
# ### MNIST score with varying rank and components
ckptdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/base_mnist_20240129-1342/checkpoints/"
df_gmm_rk = pd.read_csv(join(ckptdir, "..", "MNIST_edm_1000k_epoch_gmm_exp_var_gmm_rk.csv"))
# preprocess the dataframe to extact the rank and components
df_gmm_rk["St_residual"] = 1 - df_gmm_rk["St_EV"]
df_gmm_rk["Dt_residual"] = 1 - df_gmm_rk["Dt_EV"]
df_gmm_rk[['n_cluster', 'n_rank']] = df_gmm_rk['name'].str.extract(r'gmm_(\d+)_mode_(\d+)_rank')
df_gmm_rk['n_cluster'] = df_gmm_rk['n_cluster'].astype(float)
df_gmm_rk['n_rank'] = df_gmm_rk['n_rank'].astype(float)
#%%
res_mat_score, _ = extract_res_mats(df_gmm_rk, varname="St_residual",)
visualize_gmm_lowrank_residual(res_mat_score, 
    xvar="n_rank", huevar="n_clusters", yvar="St_residual",
    runname="MNIST mini EDM",
    savename="MNIST_miniEDM_GMM_lowrank_score_residual_x_nrank_hue_ncomp",);
visualize_gmm_lowrank_residual(res_mat_score, 
    xvar="n_clusters", huevar="n_rank", yvar="St_residual", 
    runname="MNIST mini EDM",
    savename="MNIST_miniEDM_GMM_lowrank_score_residual_x_ncomp_hue_nrank",);
visualize_gmm_lowrank_residual_heatmap_separate(res_mat_score,
        xvar="n_rank", yvar="n_clusters", plotvar="St_residual",
        runname="MNIST mini EDM",
        savename="MNIST_miniEDM_GMM_lowrank_score_residual_heatmap",);
#%%
res_mat_score, _ = extract_res_mats(df_gmm_rk, varname="St_residual",)
visualize_gmm_lowrank_residual(res_mat_score, 
    xvar="n_rank", huevar="n_clusters", yvar="St_residual",
    sigmas=[1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, 3.0e+01,
       4.0e+01, 8.0e+01], nrowcols=(2, 4),figsize=(19, 8),
    runname="MNIST mini EDM",
    savename="MNIST_miniEDM_GMM_lowrank_score_residual_x_nrank_hue_ncomp_largesigma",);

#%%
res_mat_Dt, _ = extract_res_mats(df_gmm_rk, varname="Dt_residual",)
visualize_gmm_lowrank_residual(res_mat_Dt, 
    xvar="n_rank", huevar="n_clusters", yvar="Dt_residual",
    runname="MNIST mini EDM",
    savename="MNIST_miniEDM_GMM_lowrank_denoiser_residual_x_nrank_hue_ncomp",);
visualize_gmm_lowrank_residual(res_mat_Dt, 
    xvar="n_clusters", huevar="n_rank", yvar="Dt_residual", 
    runname="MNIST mini EDM",
    savename="MNIST_miniEDM_GMM_lowrank_denoiser_residual_x_ncomp_hue_nrank",);
visualize_gmm_lowrank_residual_heatmap_separate(res_mat_Dt,
        xvar="n_rank", yvar="n_clusters", plotvar="Dt_residual",
        runname="MNIST mini EDM",
        savename="MNIST_miniEDM_GMM_lowrank_denoiser_residual_heatmap",);






#%% [markdown]
# ### CIFAR10 score with varying rank and components
tabdir = "/n/home12/binxuwang/Github/DiffusionMemorization/Tables"
df_gmm_rk = pd.read_csv(join(tabdir, "cifar10_uncond_edm_vp_pretrained_epoch_gmm_exp_var_gmm_rk.csv"))
# preprocess the dataframe to extact the rank and components
df_gmm_rk["St_residual"] = 1 - df_gmm_rk["St_EV"]
df_gmm_rk["Dt_residual"] = 1 - df_gmm_rk["Dt_EV"]
df_gmm_rk[['n_cluster', 'n_rank']] = df_gmm_rk['name'].str.extract(r'gmm_(\d+)_mode_(\d+)_rank')
df_gmm_rk['n_cluster'] = df_gmm_rk['n_cluster'].astype(float)
df_gmm_rk['n_rank'] = df_gmm_rk['n_rank'].astype(float)
#%%
res_mat_score, res_mat_score_pivot = extract_res_mats(df_gmm_rk, varname="St_residual",)
visualize_gmm_lowrank_residual(res_mat_score, 
    xvar="n_rank", huevar="n_clusters", yvar="St_residual",
    runname="CIFAR10 uncond EDM pretrained",
    savename="CIFAR_uncond_EDM_GMM_lowrank_score_residual_x_nrank_hue_ncomp",);
visualize_gmm_lowrank_residual(res_mat_score, 
    xvar="n_clusters", huevar="n_rank", yvar="St_residual", 
    runname="CIFAR10 uncond EDM pretrained",
    savename="CIFAR_uncond_EDM_GMM_lowrank_score_residual_x_ncomp_hue_nrank",);
visualize_gmm_lowrank_residual_heatmap_separate(res_mat_score,
        xvar="n_rank", yvar="n_clusters", plotvar="St_residual",
        runname="CIFAR10 uncond EDM pretrained",
        savename="CIFAR_uncond_GMM_lowrank_score_residual_heatmap",);
#%%
res_mat_score, _ = extract_res_mats(df_gmm_rk, varname="St_residual",)
visualize_gmm_lowrank_residual(res_mat_score, 
    xvar="n_rank", huevar="n_clusters", yvar="St_residual",
    sigmas=[1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, 3.0e+01,
       4.0e+01, 8.0e+01], nrowcols=(2, 4),figsize=(19, 8),
    runname="CIFAR10 uncond EDM pretrained",
    savename="CIFAR_uncond_GMM_lowrank_score_residual_x_nrank_hue_ncomp_largesigma",);

#%%
res_mat_Dt, res_mat_Dt_pivot = extract_res_mats(df_gmm_rk, varname="Dt_residual",)
visualize_gmm_lowrank_residual(res_mat_Dt, 
    xvar="n_rank", huevar="n_clusters", yvar="Dt_residual",
    runname="CIFAR10 uncond EDM pretrained",
    savename="CIFAR_uncond_GMM_lowrank_denoiser_residual_x_nrank_hue_ncomp",);
visualize_gmm_lowrank_residual(res_mat_Dt, 
    xvar="n_clusters", huevar="n_rank", yvar="Dt_residual", 
    runname="CIFAR10 uncond EDM pretrained",
    savename="CIFAR_uncond_GMM_lowrank_denoiser_residual_x_ncomp_hue_nrank",);
visualize_gmm_lowrank_residual_heatmap_separate(res_mat_Dt,
        xvar="n_rank", yvar="n_clusters", plotvar="Dt_residual",
        runname="CIFAR10 uncond EDM pretrained",
        savename="CIFAR_uncond_GMM_lowrank_denoiser_residual_heatmap",);   


 


#%%



#%% Scratch zone 
#%%
sigma = 1.0
lineplot_with_log_color_scale(res_mat_score[1.0], "n_clusters", "residual", "n_rank", 
                              cmap="turbo", ax=None, 
                              lw=1.5, marker="o", markersize=5, alpha=0.4)
plt.title(f"Score Residual of GMM MNIST Dataset | sigma={sigma:.2f}")
plt.ylabel("Score EV Residual")
plt.xlabel("Number of Modes")
plt.legend(title='n_rank')
plt.show()



#%%
# sigma = 1.5
# runname = "CIFAR10 uncond EDM pretrained"
# norm_scale = LogNorm(vmin=res_mat_pivot_all_sigma[sigma].min().min(), 
#                     vmax=res_mat_pivot_all_sigma[sigma].min().max())
# plt.figure(figsize=(10, 8))
# sns.heatmap(res_mat_pivot_all_sigma[sigma], annot=True, fmt=".2f", 
#             cmap="YlGnBu", )#norm=norm_scale)
# plt.title(f"Score EV Residual Matrix of {runname} | sigma=%.2f" % sigma)
# plt.ylabel("Number of Modes")
# plt.xlabel("Rank of Mode")
# # Increase the number of ticks
# cbar = plt.gca().collections[0].colorbar
# cbar.locator = MaxNLocator(nbins='auto')  # 'auto' can be replaced with a specific number for more control
# cbar.update_ticks()
# # saveallforms(figsumdir, f"MNIST_GMM_rank_residual_heatmap_sigma{sigma}")
# plt.show()

#%%
# # import LogNorm
from matplotlib.colors import LogNorm
for sigma in [2]:#df_gmm_rk.sigma.unique():
    # Create a heatmap
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    res_mat_pivot = res_mat.pivot_table(index="n_clusters", columns="n_rank",
                                        values="residual", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=res_mat, x="n_clusters", y="residual", 
                 hue="n_rank", palette="turbo", norm=LogNorm(), vmin=None, vmax=None, 
                 lw=1.5, marker="o", markersize=5, alpha=0.4)
    plt.yscale("log")
    plt.xscale("log")
    ylim = plt.ylim()
    plt.ylim([None, min(1, ylim[1])])
    plt.title("Score Residual of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Score EV Residual")
    plt.xlabel("Number of Modes")
    
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=res_mat, x="n_rank", y="residual", 
                 hue="n_clusters", palette="turbo", 
                 lw=1.5, marker="o", markersize=5, alpha=0.4)
    plt.yscale("log")
    plt.xscale("log")
    ylim = plt.ylim()
    plt.ylim([None, min(1, ylim[1])])
    plt.title("Score Residual of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Score EV Residual")
    plt.xlabel("rank of each Gaussian mode")
    raise ValueError("Stop here")



#%%

nrow, ncol = 2, 5
figh, axs = plt.subplots(nrow, ncol, figsize=(22.5, 8))
axs = axs.flatten()
for i, sigma in enumerate([1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ]):
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    lineplot_with_log_color_scale(res_mat, "n_clusters", "residual", "n_rank", 
                                cmap="turbo", ax=axs[i], legend=False,
                                lw=1.5, marker="o", markersize=5, alpha=0.4)
    axs[i].set_title(f"sigma={sigma:.2f}")
    axs[i].set_ylabel("Score EV Residual")
    axs[i].set_xlabel("Number of Modes")
    if i == (ncol * nrow - 1):
        axs[i].legend(title='n_rank')
plt.suptitle("Score Residual of GMM MNIST Dataset | Varying Sigma")
plt.tight_layout()
plt.show()


nrow, ncol = 2, 5
figh, axs = plt.subplots(nrow, ncol, figsize=(22.5, 8))
axs = axs.flatten()
for i, sigma in enumerate([1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ]):
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    lineplot_with_log_color_scale(res_mat, "n_rank", "residual", "n_clusters", 
                                cmap="turbo", ax=axs[i], legend=False,
                                lw=1.5, marker="o", markersize=5, alpha=0.4)
    axs[i].set_title(f"sigma={sigma:.2f}")
    axs[i].set_ylabel("Score EV Residual")
    axs[i].set_xlabel("Rank of Mode")
    if i == (ncol * nrow - 1):
        axs[i].legend(title='GMM components')
plt.suptitle("Score Residual of GMM MNIST Dataset | Varying Sigma")
plt.tight_layout()
plt.show()


# %%
for sigma in df_gmm_rk.sigma.unique():
    # Create a heatmap
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    res_mat_pivot = res_mat.pivot_table(index="n_clusters", columns="n_rank",
                                        values="residual", aggfunc="mean")
    plt.figure(figsize=(10, 8))
    sns.heatmap(res_mat_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Residual Matrix of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Number of Modes")
    plt.xlabel("Rank")
    saveallforms(figsumdir, f"MNIST_GMM_rank_residual_heatmap_sigma{sigma}")
    plt.show()

# %%
# alterantive heatmap with scientific notation
for sigma in df_gmm_rk.sigma.unique():
    # Create a heatmap
    res_mat = []
    for n_clusters in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for n_rank in [8, 16, 32, 64, 96, 128, 256, 512, 768, 1024]:
            res = 1 - df_gmm_rk[(df_gmm_rk["name"] == f"gmm_{n_clusters}_mode_{n_rank}_rank") & 
                                (df_gmm_rk.sigma == sigma)]["St_EV"].values
            res_mat.append({"n_clusters": n_clusters, "n_rank": n_rank, "residual": res[0]})
    res_mat = pd.DataFrame(res_mat)
    res_mat_pivot = res_mat.pivot_table(index="n_clusters", columns="n_rank",
                                        values="residual", aggfunc="mean")
    plt.figure(figsize=(10, 8))
    sns.heatmap(res_mat_pivot, annot=True, fmt=".1e", cmap="YlGnBu")
    plt.title("Residual Matrix of GMM MNIST Dataset | sigma=%.2f" % sigma)
    plt.ylabel("Number of Modes")
    plt.xlabel("Rank")
    plt.show()
# %%
df_gmm_rk.n_rank.unique()

# %%
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_gmm_rk[(df_gmm_rk.n_rank == 1024)],
            x="n_cluster", y="St_residual", hue="sigma", 
            palette="RdYlBu", lw=1.5, marker="o", markersize=5, alpha=0.4)
plt.yscale("log")
plt.xscale("log")
plt.title("Score residual with Varying Number of Modes | Rank=1024")
plt.show()