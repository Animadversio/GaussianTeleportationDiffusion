
#%%
import sys
from os.path import join
import matplotlib.pyplot as plt
import torch 
import numpy as np
import json
sys.path.append("../")
from gaussian_teleport.utils.plot_utils import saveallforms
from gaussian_teleport.utils.montage_utils import crop_all_from_montage, make_grid_np

def compute_sigma_steps(num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, device="cpu"):
    # Adjust noise levels based on what's supported by the network.
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max)
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    return t_steps

#%%
imgdir = r"/Users/binxuwang/OneDrive - Harvard University/Manuscript_DiffusionLinear/Figures/Teaser"
fns = ["RND000_denoised_traj_delta.png",
      "RND000_denoised_traj_edm.png",
      "RND000_denoised_traj_gauss.png",]

pattern = "RND000_denoised_traj_%s.png"
labels = ["delta", "edm", "gauss"]
idxs = [0, 5, 10, 15, 20, 25, 30, 38]
for label in labels:
    mtg = plt.imread(join(imgdir, pattern % label))
    img_traj = crop_all_from_montage(mtg, 39, imgsize=64, pad=2)
    new_mtg = make_grid_np([img_traj[i] for i in idxs], nrow=len(idxs), padding=0)
    plt.imsave(join(imgdir, "RND000_denoised_traj_%s_export.png" % label), new_mtg)
    # plt.imshow(new_mtg)
    # plt.show()
t_steps = compute_sigma_steps()
json.dump([t_steps[i].item() for i in idxs], open(join(imgdir, "t_steps.json"), "w"))
