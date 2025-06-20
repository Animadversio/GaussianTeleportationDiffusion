import sys
sys.path.append("/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Github/edm")
# torch_utils is needed from this path. 
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionMemorization")
import json
from tqdm import tqdm, trange
import re 
import glob
import os
from os.path import join
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from gaussian_teleport.edm_utils import edm_sampler
from gaussian_teleport.edm.dataset import ImageFolderDataset


from gaussian_teleport.analytical_score_lib import mean_isotropic_score, Gaussian_score, delta_GMM_score
from gaussian_teleport.analytical_score_lib import explained_var_vec
from gaussian_teleport.analytical_score_lib import sample_Xt_batch, sample_Xt_batch
from gaussian_teleport.gaussian_mixture_lib import gaussian_mixture_score_batch_sigma_torch, \
    gaussian_mixture_lowrank_score_batch_sigma_torch, compute_cluster

train_root = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Github/edm/training-runs"
def load_edm_model(ckptdir, ckpt_idx=-1, train_root=train_root, return_epoch=False):
    ckpt_list = glob.glob(join(train_root, ckptdir, "*.pkl"))
    ckpt_list = sorted(ckpt_list)
    ckpt_path = ckpt_list[ckpt_idx]
    epoch = int(re.findall(r'-(\d+).pkl', ckpt_path)[-1])
    print(f"Loading {ckpt_idx}th ckpt", ckpt_path)
    print("Epoch ", epoch)
    with open(ckpt_path, 'rb') as f:
        net = pkl.load(f)['ema'].to(device)
    if return_epoch:
        return net, epoch
    else:
        return net


def load_stats(ckptdir, train_root=train_root):
    train_stats = []
    with open(join(train_root, ckptdir, "stats.jsonl")) as f:
        for line in tqdm(f):
            train_stats.append(json.loads(line))
    return pd.DataFrame(train_stats)

device = 'cuda'
dataroot = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Github/edm/datasets"
tabdir = "/n/home12/binxuwang/Github/DiffusionMemorization/Tables"
dataset_name = "ffhq"
train_run_name = "FFHQ_edm_35k"
ckptname = "00020-ffhq-64x64-uncond-ddpmpp-edm-gpus4-batch256-fp32"
n_clusters_list = [1, 2, 5, 10, 20, ]

dataset_name = "afhq"
train_run_name = "AFHQ_edm_35k"
ckptname = "00019-afhqv2-64x64-uncond-ddpmpp-edm-gpus4-batch256-fp32"
n_clusters_list = [1, 2, 5, 10, 20, ]

dataset_name = "cifar10"
train_run_name = "cifar10_uncond_edm_5k"
ckptname = "00032-cifar10-32x32-uncond-ddpmpp-edm-gpus4-batch256-fp32"
n_clusters_list = [1, 2, 5, 10, 20, 50, 100, 200]

dataset_name = "cifar10"
train_run_name = "cifar10_uncond_edm_Aug_50k"
ckptname = "00036-cifar10-32x32-uncond-ddpmpp-edm-gpus4-batch256-fp32"
n_clusters_list = [1, 2, 5, 10, 20, 50, 100, 200]


dataset_name = "cifar10"
train_run_name = "cifar10_uncond_edm_noAug_50k"
ckptname = "00037-cifar10-32x32-uncond-ddpmpp-edm-gpus4-batch256-fp32"
n_clusters_list = [1, 2, 5, 10, 20, 50, 100, 200]

batch_size = 256
Nreps = 4
sigma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 80.0]

if dataset_name == "ffhq":
    dataset_afhq = ImageFolderDataset(join(dataroot, "ffhq-64x64.zip"))
elif dataset_name == "afhq":
    dataset_afhq = ImageFolderDataset(join(dataroot, "afhqv2-64x64.zip"))
elif dataset_name == "cifar10":
    dataset_afhq = ImageFolderDataset(join(dataroot, "cifar10-32x32.zip"))
else:
    raise ValueError(f"dataset_name {dataset_name} not recognized.")

# note these dataset output are in [0, 255] numpy array
Xtsr = np.stack([sample for sample, _ in dataset_afhq], axis=0) # (N, C, H, W)
Xtsr = torch.from_numpy(Xtsr)
Xtsr_norm = Xtsr / 127.5 - 1 # convention of edm model 
edm_Xmat = Xtsr_norm.view(Xtsr_norm.shape[0], -1)
edm_Xmat = edm_Xmat.to(device)
edm_Xmean = edm_Xmat.mean(dim=0)
edm_Xcov = (edm_Xmat - edm_Xmean).T @ (edm_Xmat - edm_Xmean) / edm_Xmat.shape[0]
eigvals, eigvecs = torch.linalg.eigh(edm_Xcov)
eigvals = eigvals.flip(0)
eigvecs = eigvecs.flip(1)
edm_imgshape = Xtsr.shape[1:]
edm_std_mean = (torch.trace(edm_Xcov) / edm_Xcov.shape[0]).sqrt()


print("Computing GMM clusters")
kmeans_batch = 2048
kmeans_random_seed = 42
kmeans_verbose = 0
lambda_EPS = 1E-5
Us_col = {}
mus_col = {}
Lambdas_col = {}
weights_col = {}
for n_clusters in reversed(n_clusters_list): #  50, 100, 
    kmeans, eigval_mat, eigvec_mat, freq_vec, center_mat = compute_cluster(edm_Xmat.cpu(), 
                            n_clusters=n_clusters,
                            kmeans_batch=kmeans_batch, 
                            kmeans_random_seed=kmeans_random_seed,
                            kmeans_verbose=kmeans_verbose,
                            lambda_EPS=lambda_EPS)
    Us_col[n_clusters] = eigvec_mat 
    mus_col[n_clusters] = center_mat 
    Lambdas_col[n_clusters] = eigval_mat 
    weights = freq_vec / freq_vec.sum()
    weights_col[n_clusters] = weights 
    print(f"n_clusters={n_clusters}, computed.")
    
print("GMM clusters computed.")

print("Defining score functionals")
score_func_col = {
    "mean isotropic": lambda Xt, sigma: mean_isotropic_score(Xt, edm_Xmean, sigma).cpu(), 
    "mean + std isotropic": lambda Xt, sigma: mean_isotropic_score(Xt, edm_Xmean, sigma, sigma0=edm_std_mean).cpu(), 
    "gaussian": lambda Xt, sigma: Gaussian_score(Xt, edm_Xmean, edm_Xcov, sigma).cpu(), 
    "gaussian regularize": lambda Xt, sigma: Gaussian_score(Xt, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma).cpu(), 
    "gmm delta": lambda Xt, sigma: delta_GMM_score(Xt, edm_Xmat, sigma).cpu(), 
}

ckpt_num = len(glob.glob(join(train_root, ckptname, "*.pkl")))
print("Explaining EDM score with GMM and other analytical scores")
df_col = []
for ckpt_idx in trange(ckpt_num):
    edm, epoch = load_edm_model(ckptname, ckpt_idx=ckpt_idx, return_epoch=True)
    edm.to(device).eval();
    print(f"ckpt_idx={ckpt_idx}, epoch={epoch}")
    for sigma in sigma_list:
        Xt_col = []
        score_vec_col = defaultdict(list)
        for rep in trange(Nreps, desc=f"sigma {sigma} rep"):
            Xt = sample_Xt_batch(edm_Xmat, batch_size, sigma=sigma).to(device)
            with torch.no_grad():
                edm_Dt = edm(Xt.view(-1, *edm_imgshape), torch.tensor(sigma).cuda(), None, ).detach().cpu()
            edm_Dt = edm_Dt.view(Xt.shape)
            score_edm = (edm_Dt - Xt.cpu()) / (sigma**2)
            score_vec_col["EDM"].append(score_edm)
            Xt_col.append(Xt.cpu())
            for score_name, analy_score_func in score_func_col.items():
                score_vec_col[score_name].append(analy_score_func(Xt, sigma))
            for n_clusters in reversed(n_clusters_list): #  50, 100, 
                gmm_scores = gaussian_mixture_score_batch_sigma_torch(Xt, 
                    mus_col[n_clusters].cuda(), Us_col[n_clusters].cuda(), Lambdas_col[n_clusters].cuda() + sigma**2, 
                    weights=weights_col[n_clusters].cuda()).cpu()
                score_vec_col[f"gmm_{n_clusters}_mode"].append(gmm_scores)
                torch.cuda.empty_cache()
        
        Xt_all = torch.cat(Xt_col, dim=0).cuda()
        for score_name, score_vec_list in score_vec_col.items():
            score_vec_col[score_name] = torch.cat(score_vec_list, dim=0)
        
        torch.cuda.empty_cache()
        
        score_edm = score_vec_col["EDM"].cuda()
        edm_Dt = score_edm * (sigma**2) + Xt_all
        for score_name, score in score_vec_col.items():
            score = score.to(device)
            Dnoiser = score * (sigma**2) + Xt_all
            exp_var_vec = explained_var_vec(score_edm, score)
            exp_var_rev_vec = explained_var_vec(score, score_edm)
            exp_var_vec_Dt = explained_var_vec(edm_Dt, Dnoiser)
            exp_var_rev_vec_Dt = explained_var_vec(Dnoiser, edm_Dt)
            St_var_vec = score.pow(2).sum(dim=1)
            Dt_var_vec = Dnoiser.pow(2).sum(dim=1)
            df_col.append({"epoch": epoch, "sigma": sigma, "name": score_name, 
                        "St_EV": exp_var_vec.mean().item(), 
                        "St_EV_std": exp_var_vec.std().item(),
                        "St_EV_rev": exp_var_rev_vec.mean().item(), 
                        "St_EV_rev_std": exp_var_rev_vec.std().item(),
                        "Dt_EV": exp_var_vec_Dt.mean().item(), 
                        "Dt_EV_std": exp_var_vec_Dt.std().item(),
                        "Dt_EV_rev": exp_var_rev_vec_Dt.mean().item(),
                        "Dt_EV_rev_std": exp_var_rev_vec_Dt.std().item(),
                        "St_Var": St_var_vec.mean().item(),
                        "St_Var_std": St_var_vec.std().item(),
                        "Dt_Var": Dt_var_vec.mean().item(), 
                        "Dt_Var_std": Dt_var_vec.std().item(),})
        torch.cuda.empty_cache()
        
    df_syn = pd.DataFrame(df_col)
    print(f"Updating csv file. {ckpt_idx}th ckpt.")
    df_syn.to_csv(join(tabdir, f"{train_run_name}_epoch_gmm_exp_var_part.csv"))

df_syn = pd.DataFrame(df_col)
df_syn["St_residual"] = 1 - df_syn["St_EV"]
df_syn["St_rev_residual"] = 1 - df_syn["St_EV_rev"]
df_syn["Dt_residual"] = 1 - df_syn["Dt_EV"]
df_syn["Dt_rev_residual"] = 1 - df_syn["Dt_EV_rev"]
df_syn.to_csv(join(tabdir, f"{train_run_name}_epoch_gmm_exp_var.csv"))
df_syn.to_csv(join(train_root, ckptname, f"{train_run_name}_epoch_gmm_exp_var.csv"))

