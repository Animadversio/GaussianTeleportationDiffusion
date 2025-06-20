import torch
import os
import re
from os.path import join
import click
import sys
sys.path.append(r"/home/biw905/Github/edm")
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

def edm_x_t_traj(xT, mu, U, Lambda, sigma_ts, sigma_T=None):
    """ Compute the analytical trajectory of x_t for a given initial condition x_T

    Args:
        xT: B x ndim tensor, a batch of initial conditions
        sigma_ts: Tdim, noise level / time discretization steps
        mu: ndim, mean of the Gaussian distribution
        U: ndim x rdim, eigenvectors of the covariance matrix
        Lambda: rdim, eigenvalues of the covariance matrix

    Returns:
        xt_traj: Tdim x B x ndim, the trajectory of x_t

    edm_x_t_traj(torch.randn(2, 100), torch.randn(100), torch.randn(100, 10), torch.rand(10), torch.arange(10)).shape
    edm_x_t_traj(torch.randn(2, 100), torch.randn(100), torch.randn(100, 100), torch.rand(100), torch.arange(10)).shape

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


def edm_sampler_return(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_traj = []
    denoised_traj = []
    x_next = latents.to(torch.float64) * t_steps[0]
    x_traj.append(x_next.detach().clone())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_traj.append(x_next.detach().clone())
        denoised_traj.append(denoised.detach().clone())

    return x_next, t_steps, x_traj, denoised_traj


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


def tsr_to_mtg(images_tsr, nrow=8, padding=2):
    images_actual = (images_tsr * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return ToPILImage()(make_grid(images_actual, nrow=nrow, padding=padding))
# Load network.
# dist.print0(f'Loading network from "{network_pkl}"...')
# Other ranks follow.
# if dist.get_rank() == 0:
#     torch.distributed.barrier()



#%% hybrid sampler
#%%
# savedir = r"E:\OneDrive - Harvard University\ICML2023_DiffGeometry\Figures\ImageSpacePCA\CIFAR10"
# figdir = r"E:\OneDrive - Harvard University\NeurIPS2023_Diffusion\Figures\edm_cifar_analytical"
# figdir = r"/home/binxu/DL_Projects/edm_analy_sample/ffhq64_uncond_vp_edm"
# PCAdir = r"/home/binxu/DL_Projects/imgdataset_PCAs"
#%%
# create a cli for seeds list, batch size, skip step list
import argparse
parser = argparse.ArgumentParser(description='CLI for handling various parameters.')
parser.add_argument('--seeds_range', nargs='+', type=int, default=[0, 512], help='List of seeds.')
parser.add_argument('--max-batch-size', type=int, default=64, help='Maximum batch size.')
parser.add_argument('--skipstep-list', nargs='+', type=int, default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 20], help='List of skip steps.')
parser.add_argument('--dataset-name', type=str, default="ffhq64", help='Name of the dataset.')
parser.add_argument('--steps', type=int, default=40, help='Number of steps.')
args = parser.parse_args()

print(f'Seeds: {args.seeds_range}')
print(f'Max Batch Size: {args.max_batch_size}')
print(f'Skipstep List: {args.skipstep_list}')
print(f'Dataset Name: {args.dataset_name}')

seeds = list(range(args.seeds_range[0], args.seeds_range[1]))
max_batch_size = args.max_batch_size
skipstep_list = args.skipstep_list
dataset_name = args.dataset_name
num_steps = args.steps
# seeds = list(range(512))
# max_batch_size = 64
# skipstep_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
# dataset_name = "ffhq64"
#%%
PCAdir = r"/home/biw905/Datasets/imgdataset_PCAs"
device = 'cuda'
if dataset_name == "ffhq64":
    # FFHQ64
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl'
    figdir = r"/n/scratch3/users/b/biw905/edm_analy_sample/ffhq64_uncond_vp_edm_theory"
    data = torch.load(join(PCAdir, "ffhq64_PCA.pt"))
    assert num_steps == 40
elif dataset_name == "afhq64":
    # AFHQ64
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl'
    figdir = r"/n/scratch3/users/b/biw905/edm_analy_sample/afhqv264_uncond_vp_edm_theory"
    data = torch.load(join(PCAdir, "afhqv264_PCA.pt"))
    assert num_steps == 40
else:
    raise NotImplementedError("Only support ffhq64 and afhq64")

# load covariance and mean of the dataset
cov_eigs, V, imgmean  = data["eigval"], data["eigvec"], data["imgmean"]
# S = S.to(device)
V = V.to(device)
imgmean = imgmean.to(device)
cov_eigs = cov_eigs.to(device)
os.makedirs(figdir, exist_ok=True)
# Load Score network.
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)
#%%
# num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
# Adjust noise levels based on what's supported by the network.
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
#%%
num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
for batch_seeds in tqdm.tqdm(all_batches, unit='batch', ):
    batch_size = len(batch_seeds)
    if batch_size == 0:
        continue
    # Pick latents and labels.
    rnd = StackedRandomGenerator(device, batch_seeds)
    latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    xT_vecs = latents.flatten(1) * t_steps[0]
    x_traj_analy = edm_x_t_traj(xT_vecs, imgmean.flatten() * 2 - 1,
                                V, cov_eigs * 4,
                                t_steps.float(), sigma_T=t_steps[0].float())
    class_labels = None
    for skipstep in skipstep_list: # [0, 1, 2, 3, 4, 5, 6, 7, 8, ]:  # range(1, num_steps):
        sigma_max_skip = t_steps[skipstep]
        print(f"skipstep={skipstep}, skip to sigma_max={sigma_max_skip}")
        skip_kwargs = dict(sigma_min=0.002, sigma_max=sigma_max_skip, rho=7, num_steps=num_steps - skipstep)
        skip_latents = x_traj_analy[skipstep].reshape(batch_size,
                              net.img_channels, net.img_resolution, net.img_resolution)
        skip_latents = skip_latents / sigma_max_skip
        skip_images, _, _, _ = edm_sampler_return(net, skip_latents, class_labels,
                                            randn_like=rnd.randn_like, **skip_kwargs)
        mtg = tsr_to_mtg(skip_images, nrow=8, padding=2)
        mtg.save(join(figdir, f"rnd{batch_seeds[0]:06d}-{batch_seeds[-1]:06d}_skip{skipstep}_noise{sigma_max_skip:.0f}.png"))
        # images_actual = (skip_images * 127.5 + 128).clip(0, 255).to(torch.uint8)
        # ToPILImage()(make_grid(images_actual, nrow=8, padding=2)).show()






#%%
# def crop_all_from_montage(img, totalnum, imgsize=32, pad=2):
#     """Return all crops from a montage image"""
#     nrow, ncol = (img.shape[0] - pad) // (imgsize + pad), (img.shape[1] - pad) // (imgsize + pad)
#     imgcol = []
#     for imgid in range(totalnum):
#         ri, ci = np.unravel_index(imgid, (nrow, ncol))
#         img_crop = img[pad + (pad + imgsize) * ri:pad + imgsize + (pad + imgsize) * ri, \
#                pad + (pad + imgsize) * ci:pad + imgsize + (pad + imgsize) * ci, :]
#         imgcol.append(img_crop)
#     return imgcol
#
# #%%
#
# for skipstep in [0, 1, 2, 3, 4, 5, 6, 7, 8, ]:  # range(1, num_steps):
#     sigma_max_skip = t_steps[skipstep]
#     print(f"skip{skipstep}_noise{sigma_max_skip:.0f}")
# #%%
# croproot = "/home/binxu/DL_Projects/edm_analy_sample/cifar10_uncond_vp_edm_crops"
# for skip, sigma_max_skip in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, ],
#                                 t_steps):
#     os.makedirs(join(croproot, f"skip{skip}_noise{sigma_max_skip:.0f}"), exist_ok=True)
# #%%
# # load all mtg figures crop and save into folders
# seeds = list(range(50000))
# max_batch_size = 256
# # iterate batches
# rank_batches
# for batch_seeds in rank_batches:
#     for skip, sigma_max_skip in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, ],
#                                     t_steps):
#         assert os.path.exists(join(figdir, f"rnd{batch_seeds[0]:06d}-{batch_seeds[-1]:06d}"
#                                            f"_skip{skip}_noise{sigma_max_skip:.0f}.png"))
#         mtg_arr = plt.imread(join(figdir, f"rnd{batch_seeds[0]:06d}-{batch_seeds[-1]:06d}"
#                                              f"_skip{skip}_noise{sigma_max_skip:.0f}.png"))
#         imgcrops = crop_all_from_montage(mtg_arr, len(batch_seeds), imgsize=32, pad=2)
#         for imgid, imgcrop in enumerate(imgcrops):
#             plt.imsave(join(croproot, f"skip{skip}_noise{sigma_max_skip:.0f}", f"rnd{batch_seeds[imgid]:06d}.png"), imgcrop)
