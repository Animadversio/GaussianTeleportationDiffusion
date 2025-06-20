import torch
import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import pandas as pd
import pickle as pkl
from os.path import join
from torch_utils import distributed as dist
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from core.utils.plot_utils import saveallforms

def edm_x_t_traj(xT, mu, U, Lambda, sigma_ts, sigma_T=None):
    """

    Args:
        xT: B x ndim
        sigma_ts: Tdim
        mu: ndim
        U: ndim x rdim
        Lambda: rdim

    Returns:

    """
    if sigma_T is None:
        sigma_T = sigma_ts.max()
    xT_rel = xT - mu[None, :]  # B x ndim
    xT_coef = xT_rel @ U  # B x rdim
    if U.shape[1] < U.shape[0]:
        xT_residue = xT_rel - xT_coef @ U.T  # B x ndim
    else:
        xT_residue = None
    scaling_coef = torch.sqrt(sigma_ts[None, :] ** 2 + Lambda[:, None])  # rdim x Tdim
    scaling_coef = scaling_coef / torch.sqrt(sigma_T ** 2 + Lambda)[:, None]  # rdim x Tdim
    xt_scaled_coef = torch.einsum('br,rT->Tbr', xT_coef, scaling_coef)  # Tdim x B x rdim
    xt_traj_onmanif = torch.einsum('Tbr,rn->Tbn', xt_scaled_coef, U.T)  # Tdim x B x ndim
    if xT_residue is not None:
        residue_scaling = sigma_ts / sigma_T  # Tdim
        residue_traj = torch.einsum('bn,T->Tbn', xT_residue, residue_scaling, )  # B x Tdim x ndim
        xt_traj = residue_traj + xt_traj_onmanif + mu[None, None, ]  # B x Tdim x ndim
    else:
        xt_traj = xt_traj_onmanif + mu[None, None, ]
    return xt_traj


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def mean_cov_from_xarr_torch(xarr):
    # xarr shape [n sample, ndim]
    # get mean and covariance of the xarr
    mu = torch.mean(xarr, dim=0)
    # Covariance
    xarr_centered = xarr - mu
    cov = torch.mm(xarr_centered.T, xarr_centered) / (xarr.shape[0] - 1)
    # eigen decomposition
    Lambda, U = torch.linalg.eigh(cov)
    # uncomment the following line to assert that eigen decomposition is correct
    # assert torch.allclose(cov, U @ torch.diag(Lambda) @ U.T)
    return mu, cov, Lambda, U



#%%
from gmm_general_diffusion_lib import gaussian_mixture_score_torch, \
    gaussian_score_torch, deltaGMM_scores_torch, deltaGMM_scores_torch_batch
# score_gauss = gaussian_mixture_score_torch(x_query, mu_all[None], U_all[None], Lambda_all[None] + sigma**2)
# score_gmm = gaussian_mixture_score_torch(x_query, mu_cls, U_cls, Lambda_cls + sigma**2)
# score_delta = deltaGMM_scores_torch(xarr_all, sigma, x_query)
#%%
def edm_gaussian_score(xT, mu, U, Lambda, sigma, ):
    """
    Args:
        xT: B x ndim
        mu: ndim
        U: ndim x rdim
        Lambda: rdim
        sigma: scalar

    Returns:
        score
    """
    xT_rel = xT - mu[None, :]  # B x ndim
    xT_coef = xT_rel @ U  # B x rdim
    scaling_coef = Lambda / (sigma ** 2 + Lambda)  # rdim
    x_onmanif = (xT_coef * scaling_coef[None, :]) @ U.T  # Tdim x B x ndim
    score_x = - (xT_rel - x_onmanif) / sigma ** 2  # B x ndim
    return score_x


def edm_gaussian_1storder_score(xT, mu, U, Lambda, sigma, ):
    """
    Args:
        xT: B x ndim
        sigma_ts: Tdim
        mu: ndim
        U: ndim x rdim
        Lambda: rdim

    Returns:
        score
    """
    xT_rel = xT - mu[None, :]  # B x ndim
    x_onmanif = ((xT_rel @ U) * (Lambda[None, :] / sigma ** 2)) @ U.T # Tdim x B x ndim
    score_x = - (xT_rel - x_onmanif) / sigma ** 2  # B x ndim
    return score_x



#%%
PCAdir = r"/home/binxu/DL_Projects/imgdataset_PCAs"
figdir = r"/home/binxu/DL_Projects/edm_score_validation"
#%%
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import trange
#%%
dataset = CIFAR10(r'~/Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
xtsr = xtsr * 2 - 1  # normalize to [-1, 1] to match the DDIM model fit to CIFAR10
#%%
mu_cls = []
cov_cls = []
Lambda_cls = []
U_cls = []
weights = []
for label in trange(10):
    print(f"computing mean cov of label={label} N={(ytsr == label).sum().item()}")
    xarr = xtsr[ytsr == label, :, :].flatten(1).cuda()
    mu, cov, Lambda, U = mean_cov_from_xarr_torch(xarr)
    assert torch.allclose(cov, U @ torch.diag(Lambda) @ U.T, atol=1E-4)
    mu_cls.append(mu)
    cov_cls.append(cov)
    Lambda_cls.append(Lambda)
    U_cls.append(U)
    weights.append(xarr.shape[0])

mu_cls = torch.stack(mu_cls, axis=0)
cov_cls = torch.stack(cov_cls, axis=0)
Lambda_cls = torch.stack(Lambda_cls, axis=0)
U_cls = torch.stack(U_cls, axis=0)
# single Gaussian approximation
xarr_all = xtsr.flatten(1).cuda()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr_torch(xarr_all)
#%%
device = 'cuda'
class_idx = None
outdir = 'validation'
# Load network.
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)
#%%
# data = torch.load(join(PCAdir, "CIFAR10_pca.pt"))
# S, V, imgmean, cov_eigs = data["S"], data["V"], data["mean"], data["cov_eigs"]
# V = V.to(device)
# imgmean = imgmean.to(device)
# cov_eigs = cov_eigs.to(device)
# torch.allclose(imgmean.flatten() * 2 - 1, mu_all, atol=1E-6)
# torch.allclose(cov_eigs * 4 , Lambda_all, atol=1E-6)
#%%
num_steps = 18
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(0, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
#%%
x_next = latents
#%%
print("max eigen", Lambda_all.max().item())
print("mean eigen", Lambda_all.mean().item())
print("min eigen", Lambda_all.min().item())
residual_stats = []
data_all = {}
EPS = 1E-5
for sigma in t_steps[:]:
    sigma_t = sigma + EPS # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
    x_probe = x_next * sigma_t
    denoised = net(x_probe, sigma_t, None)
    score_xt = (denoised - x_probe) / sigma_t ** 2
    x_probe_flatten = x_probe.flatten(1)
    print("Noise Sigma", sigma_t.item())
    ss_total_vec = ((score_xt.flatten(1))**2).sum(dim=1)
    ss_total = ss_total_vec.mean()
    model_scores = {
        'gauss': gaussian_mixture_score_torch(x_probe_flatten, mu_all[None], U_all[None],
                                              Lambda_all[None] + sigma_t ** 2),
        'gmm': gaussian_mixture_score_torch(x_probe_flatten, mu_cls, U_cls, Lambda_cls + sigma_t ** 2),
        'delta': deltaGMM_scores_torch_batch(xarr_all, sigma_t, x_probe_flatten, device="cuda", batch_size=8).cuda(),
        'iso': edm_gaussian_score(x_probe_flatten, mu_all, U_all, torch.zeros_like(Lambda_all) * 4, sigma),
    }
    stats = {'sigma': sigma_t.item(), 'ss_total': ss_total.item()}
    data = {"score_nn": score_xt.flatten(1).cpu()}
    for model, score_pred in model_scores.items():
        residual_vec = ((score_xt.flatten(1) - score_pred) ** 2).sum(dim=1)
        residual = residual_vec.mean()
        ratio = residual / ss_total
        print(f"residual variance ratio of {model} model: {ratio.item()}")
        stats[f"residual_{model}"] = residual.item()
        stats[f"resid_ratio_{model}"] = ratio.item()
        data[f"score_{model}"] = score_pred.cpu()
        data[f"residual_{model}_vec"] = residual_vec.cpu()

    residual_stats.append(stats)
    data_all[sigma_t.item()] = data

residual_df = pd.DataFrame(residual_stats)
residual_df.to_csv(join(figdir, "cifar10_GMM_score_approx_residual.csv"))

pkl.dump(data_all, open(join(figdir, "cifar10_GMM_score_approx.pkl"), "wb"))
#%%
plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(Lambda_all, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(Lambda_all.mean()).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gauss"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_delta"], "-o",
             label="Score of delta GMM", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gmm"], "-o",
             label="Score of GMM", alpha=0.5)
plt.ylim([None, 1E2])
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("CIFAR10 GMM score approximation")
saveallforms(figdir, "cifar10_GMM_score_approx_residual_curve")
plt.show()
#%%

plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(Lambda_all, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(Lambda_all.mean()).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gauss"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_delta"], "-o",
             label="Score of delta GMM", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gmm"], "-o",
             label="Score of GMM", alpha=0.5)
plt.legend()
plt.xlim([-0.02, 10])
plt.ylim([1E-5, 1E2])
plt.gca().relim(visible_only=True)
plt.gca().autoscale_view(scalex=False)
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("CIFAR10 GMM score approximation")
saveallforms(figdir, "cifar10_GMM_score_approx_residual_curve_xlim")
plt.show()





#%%
from training.dataset import ImageFolderDataset
dataset = ImageFolderDataset("/home/binxu/Datasets/afhqv2-64x64.zip")
# xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0) # note this is buggy!
xtsr = np.stack([(img) for img, _ in dataset], axis=0) # note this is buggy!
xtsr = torch.from_numpy(xtsr).float() / 255.0
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
xtsr = xtsr * 2 - 1  # normalize to [-1, 1] to match the DDIM model fit to CIFAR10
#%%
# single Gaussian approximation
xarr_all = xtsr.flatten(1).cuda()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr_torch(xarr_all)
cov_all = cov_all.to("cpu")
#%%
device = 'cuda'
class_idx = None
# Load network.
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

# data = torch.load(join(PCAdir, "afhqv264_PCA.pt"))
# cov_eigs, V, imgmean = data["eigval"], data["eigvec"], data["imgmean"]
# V = V.to(device)
# imgmean = imgmean.to(device)
# cov_eigs = cov_eigs.to(device)
#%%
num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(0, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
#%%
x_next = latents
#%%
print("max eigen", Lambda_all.max().item())
print("mean eigen", Lambda_all.mean().item())
print("min eigen", Lambda_all.min().item())
residual_stats = []
data_all = {}
EPS = 1E-5
for sigma in t_steps[:]:
    sigma_t = sigma + EPS # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
    x_probe = x_next * sigma_t
    denoised = net(x_probe, sigma_t, None)
    score_xt = (denoised - x_probe) / sigma_t ** 2
    x_probe_flatten = x_probe.flatten(1)
    print("Noise Sigma", sigma_t.item())
    ss_total_vec = ((score_xt.flatten(1))**2).sum(dim=1)
    ss_total = ss_total_vec.mean()
    model_scores = {
        'gauss': gaussian_mixture_score_torch(x_probe_flatten, mu_all[None], U_all[None],
                                              Lambda_all[None] + sigma_t ** 2),
        # 'gmm': gaussian_mixture_score_torch(x_probe_flatten, mu_cls, U_cls, Lambda_cls + sigma ** 2),
        'delta': deltaGMM_scores_torch_batch(xarr_all, sigma_t, x_probe_flatten, device="cuda", batch_size=4).cuda(),
        'iso': edm_gaussian_score(x_probe_flatten, mu_all, U_all, torch.zeros_like(Lambda_all), sigma_t),
    }
    stats = {'sigma': sigma.item(), 'ss_total': ss_total.item()}
    data = {"score_nn": score_xt.flatten(1).cpu()}
    for model, score_pred in model_scores.items():
        residual_vec = ((score_xt.flatten(1) - score_pred) ** 2).sum(dim=1)
        residual = residual_vec.mean()
        ratio = residual / ss_total
        print(f"residual variance ratio of {model} model: {ratio.item()}")
        stats[f"residual_{model}"] = residual.item()
        stats[f"resid_ratio_{model}"] = ratio.item()
        data[f"score_{model}"] = score_pred.cpu()
        data[f"residual_{model}_vec"] = residual_vec.cpu()

    residual_stats.append(stats)
    data_all[sigma_t.item()] = data

residual_df = pd.DataFrame(residual_stats)
residual_df.to_csv(join(figdir, "AFHQv264_GMM_score_approx_residual.csv"))

pkl.dump(data_all, open(join(figdir, "AFHQv264_GMM_score_approx.pkl"), "wb"))
#%%

plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(Lambda_all, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(Lambda_all.mean()).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gauss"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gmm"], "-o",
#              label="Score of GMM", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_delta"], "-o",
             label="Score of delta GMM", alpha=0.5)
plt.ylim([None, 1E2])
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("AFHQv2 GMM score approximation")
saveallforms(figdir, "AFHQv2_GMM_score_approx_residual_curve")
plt.show()

#%%
plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(Lambda_all, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(Lambda_all.mean()).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gauss"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gmm"], "-o",
#              label="Score of GMM", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_delta"], "-o",
             label="Score of delta GMM", alpha=0.5)
plt.ylim([1E-4, 1E2])
plt.xlim([-0.02, 10])
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("AFHQv2 GMM score approximation")
saveallforms(figdir, "AFHQv2_GMM_score_approx_residual_curve_xlim")
plt.show()



#%%
from training.dataset import ImageFolderDataset
dataset = ImageFolderDataset("/home/binxu/Datasets/ffhq-64x64.zip")
# load whole MNIST into a single tensor
# xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0)
xtsr = np.stack([(img) for img, _ in dataset], axis=0) # note this is buggy!
xtsr = torch.from_numpy(xtsr).float() / 255.0
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
xtsr = xtsr * 2 - 1  # normalize to [-1, 1] to match the DDIM model fit to CIFAR10
#%%
# single Gaussian approximation
xarr_all = xtsr.flatten(1).cuda()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr_torch(xarr_all)
cov_all = cov_all.to("cpu")

#%%
device = 'cuda'
class_idx = None
# Load network.
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

# data = torch.load(join(PCAdir, "ffhq64_PCA.pt"))
# cov_eigs, V, imgmean = data["eigval"], data["eigvec"], data["imgmean"]
# V = V.to(device)
# imgmean = imgmean.to(device)
# cov_eigs = cov_eigs.to(device)
#%%
num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(0, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
x_next = latents
#%%
print("max eigen", Lambda_all.max().item())
print("mean eigen", Lambda_all.mean().item())
print("min eigen", Lambda_all.min().item())
residual_stats = []
data_all = {}
EPS = 1E-5
for sigma in t_steps[:]:
    sigma_t = sigma + EPS # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
    x_probe = x_next * sigma_t
    denoised = net(x_probe, sigma_t, None)
    score_xt = (denoised - x_probe) / sigma_t ** 2
    x_probe_flatten = x_probe.flatten(1)
    print("Noise Sigma", sigma_t.item())
    ss_total_vec = ((score_xt.flatten(1))**2).sum(dim=1)
    ss_total = ss_total_vec.mean()
    model_scores = {
        'gauss': gaussian_mixture_score_torch(x_probe_flatten, mu_all[None], U_all[None],
                                              Lambda_all[None] + sigma_t ** 2),
        # 'gmm': gaussian_mixture_score_torch(x_probe_flatten, mu_cls, U_cls, Lambda_cls + sigma ** 2),
        'delta': deltaGMM_scores_torch_batch(xarr_all, sigma_t, x_probe_flatten, device="cuda", batch_size=1).cuda(),
        'iso': edm_gaussian_score(x_probe_flatten, mu_all, U_all, torch.zeros_like(Lambda_all), sigma_t),
    }
    stats = {'sigma': sigma.item(), 'ss_total': ss_total.item()}
    data = {"score_nn": score_xt.flatten(1).cpu()}
    for model, score_pred in model_scores.items():
        residual_vec = ((score_xt.flatten(1) - score_pred) ** 2).sum(dim=1)
        residual = residual_vec.mean()
        ratio = residual / ss_total
        print(f"residual variance ratio of {model} model: {ratio.item()}")
        stats[f"residual_{model}"] = residual.item()
        stats[f"resid_ratio_{model}"] = ratio.item()
        data[f"score_{model}"] = score_pred.cpu()
        data[f"residual_{model}_vec"] = residual_vec.cpu()

    residual_stats.append(stats)
    data_all[sigma_t.item()] = data

residual_df = pd.DataFrame(residual_stats)
residual_df.to_csv(join(figdir, "FFHQ64_GMM_score_approx_residual.csv"))

pkl.dump(data_all, open(join(figdir, "FFHQ64_GMM_score_approx.pkl"), "wb"))
#%%

plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(Lambda_all, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(Lambda_all.mean()).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gauss"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gmm"], "-o",
#              label="Score of GMM", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_delta"], "-o",
             label="Score of delta GMM", alpha=0.5)
plt.ylim([None, 1E2])
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("FFHQ GMM score approximation")
saveallforms(figdir, "FFHQ_GMM_score_approx_residual_curve")
plt.show()

#%%
plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(Lambda_all, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(Lambda_all.mean()).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gauss"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gmm"], "-o",
#              label="Score of GMM", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_delta"], "-o",
             label="Score of delta GMM", alpha=0.5)
plt.ylim([1E-4, 1E2])
plt.xlim([-0.02, 10])
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("FFHQ GMM score approximation")
saveallforms(figdir, "FFHQ_GMM_score_approx_residual_curve_xlim")
plt.show()




#%% Synopsis
from os.path import join
import pandas as pd
figdir = r"/home/binxu/DL_Projects/edm_score_validation"
# load the csv
for dataset_name in ["cifar10",
                     "AFHQv264",
                     "FFHQ64"]:
    residual_df = pd.read_csv(join(figdir, f"{dataset_name}_GMM_score_approx_residual.csv"))
    print(dataset_name)
    msk = (residual_df["resid_ratio_gauss"] < 0.01)
    print(f"Gaussian explains > 0.99 from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}",)
    msk = (residual_df["resid_ratio_gauss"] < residual_df["resid_ratio_delta"])
    print(f"Gaussian better than Point cloud (delta GMM) from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}",)
    msk = (residual_df["resid_ratio_gauss"] > residual_df["resid_ratio_delta"])
    print( f"Point cloud (delta GMM) better than Gaussian from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}", )
    msk = (residual_df["resid_ratio_gauss"] < residual_df["resid_ratio_iso"])
    print(f"Gaussian better than data center from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}",)
    if "resid_ratio_gmm" in residual_df:
        msk = (residual_df["resid_ratio_gmm"] < residual_df["resid_ratio_gauss"])
        print(f"Gaussian better than GMM from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}",)

#%%%
# redirect output to a file
import sys
from contextlib import redirect_stdout
with open(join(figdir, "score_validation_synopsis.txt"), "w") as f:
    with redirect_stdout(f):
        for dataset_name in ["cifar10",
                             "AFHQv264",
                             "FFHQ64"]:
            residual_df = pd.read_csv(join(figdir, f"{dataset_name}_GMM_score_approx_residual.csv"))
            print(dataset_name)
            msk = (residual_df["resid_ratio_gauss"] < 0.01)
            print(f"Gaussian explains > 0.99 from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}", )
            msk = (residual_df["resid_ratio_gauss"] < residual_df["resid_ratio_delta"])
            print(
                f"Gaussian better than Point cloud (delta GMM) from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}", )
            msk = (residual_df["resid_ratio_gauss"] > residual_df["resid_ratio_delta"])
            print(
                f"Point cloud (delta GMM) better than Gaussian from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}", )
            msk = (residual_df["resid_ratio_gauss"] < residual_df["resid_ratio_iso"])
            print(
                f"Gaussian better than data center from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}", )
            if "resid_ratio_gmm" in residual_df:
                msk = (residual_df["resid_ratio_gmm"] < residual_df["resid_ratio_gauss"])
                print(
                    f"Gaussian better than GMM from {residual_df[msk].sigma.min()} to {residual_df[msk].sigma.max()}", )

