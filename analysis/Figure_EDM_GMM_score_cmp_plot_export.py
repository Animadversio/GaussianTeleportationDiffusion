"""
This script is used to visualize the approximation error of EDM score by GMM score. 
At various training steps and with different number of modes and components. 
It assumes the computation has been done and the results are stored in the csv files.
with the key plotting function `visualize_train_run_score_cmp`.
"""
# %% [markdown]
# ## GMM vs EDM 
# %%
import os
import sys
from os.path import join
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("../")
from gaussian_teleport.utils.plot_utils import saveallforms
# set pandas display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#%%
from matplotlib.ticker import ScalarFormatter
epochfmt = ScalarFormatter()
epochfmt.set_powerlimits((-3,4))  # Or whatever the limits are . . .
def visualize_train_run_score_cmp(df_syn, plot_var="St_residual", palette="turbo", 
             hue_order=["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                        "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"], 
             sigmas=[1.0e-02, 5.0e-02, 1.0e-01, 5.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 5.0e+00, 1.0e+01, 2.0e+01, ],
             figsize=(22.5, 8), nrowcols=(2, 5), 
            train_run_name="CIFAR10 EDM 5k epochs initial", 
            savename="cifar10_mini_edm_gmm_score_residual_ev_initial_epochs"):
    
    if plot_var == "St_residual":
        cmp_variable = "score"
    elif plot_var == "Dt_residual":
        cmp_variable = "denoiser"
    else:
        cmp_variable = plot_var
    nrow, ncol = nrowcols
    figh, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.flatten()
    for i, sigma in enumerate(sigmas):
        df_syn_sigma = df_syn[(df_syn.sigma == sigma)]
        sns.lineplot(data=df_syn_sigma, x="epoch", y=plot_var, hue="name", 
                    hue_order=hue_order, palette=palette, ax=axs[i], 
                    marker="o", markersize=3, lw=2.0, alpha=0.7)
        axs[i].set_yscale("log")
        axs[i].set_ylim(None, 1)
        axs[i].set_ylabel(f"{cmp_variable} residual EV", fontsize=18)
        axs[i].set_xlabel(f"epoch", fontsize=18)
        axs[i].tick_params(axis='both', which='major', labelsize=18)
        axs[i].tick_params(axis='both', which='minor', labelsize=16)
        axs[i].xaxis.set_major_formatter(epochfmt)
        axs[i].get_xaxis().get_offset_text().set_size(16)
        axs[i].get_yaxis().get_offset_text().set_size(16)
        axs[i].set_title(f"sigma={sigma}", fontsize=20, y=0.94)
        # remove ylabel when not at the leftmost column
        if i % ncol != 0:
            axs[i].set_ylabel(None)
        # remove xlabel when not at the bottom row
        if i < (nrow - 1) * ncol:
            axs[i].set_xlabel(None)
        if i < (nrow * ncol - 1):
            # remove legend when not at the final panel 
            axs[i].legend().remove()
        else:
            # move the legend to outside of the plot
            axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16,
                          title="Score approximation", title_fontsize=18)
    plt.suptitle(f"Residual explained variance of EDM {cmp_variable} by Gaussian and GMM {cmp_variable}s ({train_run_name})", fontsize=24)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # plt.tight_layout()
    saveallforms(figsumdir, savename, figh, ["pdf", "png"])
    plt.show()
    return figh, axs


# %%
rootdir = "../" # e.g. "/Users/binxuwang/Github/GaussianTeleportationDiffusion"
tabdir = join(rootdir, "Tables")
figroot = join(rootdir, "Figures")
figsumdir = join(figroot, "GMM_EDM_training_summary")
os.makedirs(figsumdir, exist_ok=True)

# %%
# %% [markdown]
# ### MNIST EDM training process vs GMM scores
# %% [markdown]
# #### Early Phase MNIST
df_syn_mnst = pd.read_csv(join(tabdir, "MNIST_edm_25k_epoch_gmm_exp_var.csv"))
df_syn_mnst.name.unique()
mnist_score_names = ["mean isotropic", "gaussian", 'gmm_2_mode', 
                'gmm_5_mode', 'gmm_10_mode', 'gmm_20_mode', 'gmm_50_mode', 
                'gmm_100_mode', 'gmm_200_mode','gmm_500_mode', "gmm delta"]
palette = "turbo" # "viridis_r"# "RdYlBu" "plasma" #"magma" # cividis" # "Spectral"
visualize_train_run_score_cmp(df_syn_mnst, plot_var="St_residual", 
                              hue_order=mnist_score_names, palette=palette,
                              train_run_name="MNIST EDM 25k epochs early phase",
                              savename="mnist_mini_edm_gmm_score_residual_ev_initial_epochs");

visualize_train_run_score_cmp(df_syn_mnst, plot_var="Dt_residual", 
                              hue_order=mnist_score_names, palette=palette, 
                              train_run_name="MNIST EDM 25k epochs early phase",
                              savename="mnist_mini_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### Late Phase MNIST
df_syn_mnst_lt = pd.read_csv(join(tabdir, "MNIST_edm_1000k_epoch_gmm_exp_var.csv"))

mnist_score_names = ["mean isotropic", "gaussian", 'gmm_2_mode', 
                'gmm_5_mode', 'gmm_10_mode', 'gmm_20_mode', 'gmm_50_mode', 
                'gmm_100_mode', 'gmm_200_mode','gmm_500_mode', "gmm delta"]

visualize_train_run_score_cmp(df_syn_mnst_lt, plot_var="St_residual", 
                              hue_order=mnist_score_names, 
                              train_run_name="MNIST EDM 1000k epochs early phase",
                              savename="mnist_mini_edm_gmm_score_residual_ev_late_epochs");

visualize_train_run_score_cmp(df_syn_mnst_lt, plot_var="Dt_residual", 
                              hue_order=mnist_score_names, 
                              train_run_name="MNIST EDM 1000k epochs early phase",
                              savename="mnist_mini_edm_gmm_denoiser_residual_ev_late_epochs");


# %% [markdown]
# ### CIFAR10 EDM training process vs GMM scores

# %% [markdown]
# #### CIFAR Early phase ckpt stats
df_syn_cf = pd.read_csv(join(tabdir, "edm_50k_epoch_gmm_exp_var.csv"))
# df_syn_cf.sigma.unique()
cifar_score_names = ["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"]
visualize_train_run_score_cmp(df_syn_cf, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 50k epochs early phase",
                              savename="cifar10_mini_edm_gmm_score_residual_ev_initial_epochs");

visualize_train_run_score_cmp(df_syn_cf, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 50k epochs early phase",
                              savename="cifar10_mini_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### Late phase ckpt CIFAR10
df_syn_cf_lt = pd.read_csv(join(tabdir, "edm_365k_epoch_gmm_exp_var.csv"))
# df_syn_cf.sigma.unique()
cifar_score_names = ["mean isotropic", "gaussian", "gmm_2_mode", "gmm_5_mode", 
                       "gmm_10_mode", "gmm_20_mode", "gmm_50_mode", "gmm_100_mode", "gmm delta"]
visualize_train_run_score_cmp(df_syn_cf_lt, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 365k epochs late phase",
                              savename="cifar10_mini_edm_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_cf_lt, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 EDM 365k epochs late phase",
                              savename="cifar10_mini_edm_gmm_denoiser_residual_ev_late_epochs");


# %% [markdown]
# ### CIFAR EDM trained early
df_syn_cifar2 = pd.read_csv(join(tabdir, "cifar10_uncond_edm_5k_epoch_gmm_exp_var.csv"))
cifar_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 'gmm_10_mode', 
                     'gmm_20_mode', 'gmm_50_mode', 'gmm_100_mode', 'gmm_200_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_cifar2, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM 5k epochs early phase",
                              savename="cifar10_edm_gmm_score_residual_ev_early_epochs");
visualize_train_run_score_cmp(df_syn_cifar2, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM 5k epochs early phase",
                              savename="cifar10_edm_gmm_denoiser_residual_ev_early_epochs");

# %% [markdown]
# ### CIFAR EDM trained Late, 50k epochs, Augmented
df_syn_cifar3 = pd.read_csv(join(tabdir, "cifar10_uncond_edm_Aug_50k_epoch_gmm_exp_var.csv"))
cifar_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 'gmm_10_mode', 
                     'gmm_20_mode', 'gmm_50_mode', 'gmm_100_mode', 'gmm_200_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_cifar3, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM (Aug) 50k epochs late phase",
                              savename="cifar10_edm_Aug_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_cifar3, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM (Aug) 50k epochs late phase",
                              savename="cifar10_edm_Aug_gmm_denoiser_residual_ev_late_epochs");
#%%
# ### CIFAR EDM trained late, 50k epochs no Augmentation
df_syn_cifar4 = pd.read_csv(join(tabdir, "cifar10_uncond_edm_noAug_50k_epoch_gmm_exp_var.csv"))
cifar_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 'gmm_10_mode', 
                     'gmm_20_mode', 'gmm_50_mode', 'gmm_100_mode', 'gmm_200_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_cifar4, plot_var="St_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM (noAug) 50k epochs late phase",
                              savename="cifar10_edm_noAug_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_cifar4, plot_var="Dt_residual", 
                              hue_order=cifar_score_names, 
                              train_run_name="CIFAR10 origin EDM (noAug) 50k epochs late phase",
                              savename="cifar10_edm_noAug_gmm_denoiser_residual_ev_late_epochs");

# %% [markdown]
# ### AFHQ 64 Dataseet early phase
df_syn_afhq = pd.read_csv(join(tabdir, "AFHQ_edm_5k_epoch_gmm_exp_var_fixed.csv"))
df_syn_afhq.name.unique()
afhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_afhq, plot_var="St_residual", 
                              hue_order=afhq_score_names, 
                              train_run_name="AFHQ origin EDM 5k epochs early phase",
                              savename="afhq_edm_gmm_score_residual_ev_initial_epochs");
visualize_train_run_score_cmp(df_syn_afhq, plot_var="Dt_residual",
                                hue_order=afhq_score_names, 
                                train_run_name="AFHQ origin EDM 5k epochs early phase",
                                savename="afhq_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### AFHQ 64 Dataseet Late training
df_syn_afhq_lt = pd.read_csv(join(tabdir, "AFHQ_edm_35k_epoch_gmm_exp_var.csv"))
afhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_afhq_lt, plot_var="St_residual", 
                              hue_order=afhq_score_names, 
                              train_run_name="AFHQ origin EDM 35k epochs late phase",
                              savename="afhq_edm_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_afhq_lt, plot_var="Dt_residual",
                                hue_order=afhq_score_names, 
                                train_run_name="AFHQ origin EDM 35k epochs late phase",
                                savename="afhq_edm_gmm_denoiser_residual_ev_late_epochs");

# %% [markdown]
# ### FFHQ dataset early phase
df_syn_ffhq = pd.read_csv(join(tabdir,"FFHQ_edm_5k_epoch_gmm_exp_var.csv"))
ffhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_ffhq, plot_var="St_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 5k epochs early phase",
                                savename="ffhq_edm_gmm_score_residual_ev_initial_epochs");
visualize_train_run_score_cmp(df_syn_ffhq, plot_var="Dt_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 5k epochs early phase",
                                savename="ffhq_edm_gmm_denoiser_residual_ev_initial_epochs");

# %% [markdown]
# #### FFHQ Late phase training
df_syn_ffhq_lt = pd.read_csv(join(tabdir,"FFHQ_edm_35k_epoch_gmm_exp_var.csv"))
ffhq_score_names = ["mean_isotropic", "gaussian", 'gmm_2_mode', 'gmm_5_mode', 
                    'gmm_10_mode', 'gmm_20_mode', "gmm delta"]
visualize_train_run_score_cmp(df_syn_ffhq_lt, plot_var="St_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 35k epochs late phase",
                                savename="ffhq_edm_gmm_score_residual_ev_late_epochs");
visualize_train_run_score_cmp(df_syn_ffhq_lt, plot_var="Dt_residual",
                                hue_order=ffhq_score_names, 
                                train_run_name="FFHQ origin EDM 35k epochs late phase",
                                savename="ffhq_edm_gmm_denoiser_residual_ev_late_epochs");

