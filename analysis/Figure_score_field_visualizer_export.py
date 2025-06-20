# %% [markdown]
# ## EDM vs Analytical Score
import os 
from os.path import join
import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from gaussian_teleport.edm_utils import get_default_config, create_edm, edm_sampler, EDM, create_model
from gaussian_teleport.analytical_score_lib import mean_isotropic_score, Gaussian_score, delta_GMM_score
from gaussian_teleport.analytical_score_lib import explained_var_vec
from gaussian_teleport.analytical_score_lib import sample_Xt_batch, sample_Xt_batch
from gaussian_teleport.gaussian_mixture_lib import gaussian_mixture_score_batch_sigma_torch, \
    gaussian_mixture_lowrank_score_batch_sigma_torch, compute_cluster
from gaussian_teleport.utils.plot_utils import saveallforms
# set pandas display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# %% [markdown]
# ### Load and Examine the EDM trained model w.r.t. scores

# %% [markdown]
# ### Util functions for 2d slicing projection

class CoordSystem:
    """Define a 2D coordinate system with two basis vectors and an origin."""
    def __init__(self, basis1, basis2, origin=None):
        # Ensure the basis vectors are normalized and orthogonal
        # if origin is None:
        #     origin = torch.zeros_like(basis1)
        self.reference = origin
        self.basis1 = basis1 / torch.norm(basis1)
        self.basis2 = basis2 / torch.norm(basis2)
        self.basis_matrix = torch.stack([self.basis1, self.basis2], dim=0)
        self.device = self.basis_matrix.device
        
    def project_vector(self, vectors):
        # Project a vector onto the basis using matrix algebra
        return (vectors.to(self.device) @ self.basis_matrix.to(vectors.dtype).T)

    def ortho_project_vector(self, vectors):
        # Project a vector onto the perpendicular space of the basis using matrix algebra
        return vectors.to(self.device) - (vectors.to(self.device) @ self.basis_matrix.to(vectors.dtype).T) @ self.basis_matrix.to(vectors.dtype)
    
    def project_points(self, points):
        # Project a set of points onto the basis
        return (points.to(self.device) - self.reference.to(points.dtype)) @ self.basis_matrix.to(points.dtype).T
    #TODO: make the casting more elegant


def orthogonal_grid(x1, x2, x3, grid_size):
    # Step 1: Find the Basis Vectors
    v1 = x2 - x1
    v = x3 - x1
    # Orthogonalize v with respect to v1 using the Gram-Schmidt process
    proj_v1_v = np.dot(v, v1) / np.dot(v1, v1) * v1
    v2 = v - proj_v1_v
    # Step 2: Normalize the Basis Vectors
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    # Step 3: Create the Grid Points
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Scaling factors for v1 and v2
            scale_v1 = i / (grid_size - 1)
            scale_v2 = j / (grid_size - 1)
            # Generate the grid point
            grid_point = x1 + scale_v1 * v1_normalized + scale_v2 * v2_normalized
            grid_points.append(grid_point)

    return np.array(grid_points)

def orthogonal_grid_torch(x1, x2, x3, grid_nums=(10, 10), 
                          x_range=(0, 1), y_range=(0, 1)):
    # Step 1: Find the Basis Vectors
    v1 = x2 - x1
    v = x3 - x1
    # Orthogonalize v with respect to v1 using the Gram-Schmidt process
    proj_v1_v = torch.dot(v, v1) / torch.dot(v1, v1) * v1
    v2 = v - proj_v1_v
    v1_norm = torch.norm(v1)
    v2_norm = torch.norm(v2)
    # Step 2: Normalize the Basis Vectors
    v1_normalized = v1 / v1_norm
    v2_normalized = v2 / v2_norm
    coordsys = CoordSystem(v1_normalized, v2_normalized, origin=x1)
    # Step 3: Create the Grid Points
    grid_vecs = []
    norm_coords = []
    plane_coords = []
    for ti in torch.linspace(x_range[0], x_range[1], grid_nums[0]):
        for tj in torch.linspace(y_range[0], y_range[1], grid_nums[1]):
            # Scaling factors for v1 and v2
            scale_v1 = ti * v1_norm
            scale_v2 = tj * v2_norm
            # Generate the grid point
            grid_vec = x1 + scale_v1 * v1_normalized + scale_v2 * v2_normalized
            grid_vecs.append(grid_vec)
            norm_coords.append([ti, tj])
            plane_coords.append([scale_v1, scale_v2])
    return torch.stack(grid_vecs), \
            torch.tensor(norm_coords), \
            torch.tensor(plane_coords), coordsys


def test_project_numerical(edm_Xmat):
    if edm_Xmat is None:
        edm_Xmat = torch.randn(3, 4096).cuda()
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        edm_Xmat[0, :], edm_Xmat[1, :], edm_Xmat[2, :], 
        grid_nums=(16, 16), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))
    # TODO Define test function for the score projection
    assert torch.allclose(coordsys.ortho_project_vector(grid_vecs) + 
                coordsys.project_vector(grid_vecs) @ coordsys.basis_matrix, 
                grid_vecs, atol=1E-5)
    assert torch.allclose(plane_coords.cuda(), coordsys.project_points(grid_vecs))
    print("Projection Numerical Test passed")

test_project_numerical(None)
# %% [markdown]
# ### Visualization pipeline function

def score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape,
                           sigmas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], titlestr="",
                           grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25),
                           figsumdir=figsumdir, savestr="", show_image=True,):
    anchors_tsr = torch.stack(anchors)
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
            *anchors, grid_nums=grid_nums, x_range=x_range, y_range=y_range)
    if show_image:
        mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=grid_nums[1])
        plt.figure(figsize=(10, 10))
        plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        saveallforms(figsumdir, f"sample_image_map_{savestr}")
        plt.show()
    for sigma in sigmas:
        edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
        score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
        score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
        score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
        score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
        # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
        # Calculate the vector field
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                            ("gmm delta", score_gmm_Xt),
                            ("gaussian", score_gaussian_Xt), 
                            # ("gaussian regularize", score_gaussian_reg_Xt),
                            ("mean isotropic", score_mean_Xt), 
                            # ("mean + std isotropic", score_mean_std_Xt), 
                            # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                            # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                            ]):
            # Create a grid for the quiver plot
            vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
            anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
            axs[i].invert_yaxis()
            axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
            axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
            axs[i].set_aspect('equal')
            axs[i].set_title(name)
        plt.suptitle(f"score vector field {titlestr}\nsigma={sigma:.2f}")
        saveallforms(figsumdir, f"score_vector_field_{savestr}_sigma_{sigma}")
        plt.show()

# %%
def score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="l2",
                           sigmas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], titlestr="",
                           grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25),
                           figsumdir=figsumdir, savestr="", show_image=True,):
    anchors_tsr = torch.stack(anchors)
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
            *anchors, grid_nums=grid_nums, x_range=x_range, y_range=y_range)
    if show_image:
        mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=grid_nums[1])
        plt.figure(figsize=(10, 10))
        plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        saveallforms(figsumdir, f"sample_image_map_{savestr}")
        plt.show()
    for sigma in sigmas:
        edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
        score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
        score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
        score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
        score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
        # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
        score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
        # Calculate the vector field
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                            ("gmm delta", score_gmm_Xt),
                            ("gaussian", score_gaussian_Xt), 
                            # ("gaussian regularize", score_gaussian_reg_Xt),
                            ("mean isotropic", score_mean_Xt), 
                            # ("mean + std isotropic", score_mean_std_Xt), 
                            # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                            # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                            ]):
            # Create a grid for the quiver plot
            anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
            axs[i].invert_yaxis()
            if magnitude == "l2":
                vec_norm = torch.norm(vector_field, dim=-1).cpu().numpy()
            elif magnitude == "proj_l2":
                vec_proj = coordsys.project_vector(vector_field)
                vec_norm = torch.norm(vec_proj, dim=-1).cpu().numpy()
            elif magnitude == "ortho_l2":
                vec_ortho = coordsys.ortho_project_vector(vector_field.double())
                vec_norm = torch.norm(vec_ortho, dim=-1).cpu().numpy()
            # axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
            im = axs[i].imshow(vec_norm.reshape(grid_nums), origin="lower", 
                          extent=[plane_coords[:, 1].min(), plane_coords[:, 1].max(), 
                                  plane_coords[:, 0].min(), plane_coords[:, 0].max(), ],
                          vmin=0, vmax=vec_norm.max())
            axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
            axs[i].invert_yaxis()
            axs[i].set_aspect('equal')
            axs[i].set_title(name)
            # add colorbar
            # divider = make_axes_locatable(axs[i])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, )
            # cbar.set_clim(0, vec_norm.max())
        plt.suptitle(f"score vector field {titlestr}\nsigma={sigma:2f} magnitude={magnitude}")
        saveallforms(figsumdir, f"score_magnitude_map_{savestr}_{magnitude}_sigma_{sigma}")
        plt.show()


#%% [markdown]
# ## More flexible functional interface to plot arbitrary score function
def score_slice_projection_functional(anchors, score_func_col, score_name2plot, edm_imgshape,
                           sigmas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], titlestr="",
                           grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25),
                           figsize=(20, 5), nrowcols=(1, 4), savefig=True, colorvector=True,
                           figsumdir=figsumdir, savestr="", show_image=True,):
    anchors_tsr = torch.stack(anchors)
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
            *anchors, grid_nums=grid_nums, x_range=x_range, y_range=y_range)
    if show_image:
        mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=grid_nums[1])
        plt.figure(figsize=(10, 10))
        plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if savefig: saveallforms(figsumdir, f"sample_image_map_{savestr}")
        plt.show()
    for sigma in sigmas:
        # edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
        # score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
        # Calculate the vector field
        score_vec_col = {}
        # score_vec_col["edm NN"] = score_edm
        for score_name in score_name2plot:
            score_func = score_func_col[score_name]
            score_vec_col[score_name] = score_func(grid_vecs, sigma)
        
        nrow, ncol = nrowcols
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
        axs = axs.flatten()
        for i, (name, vector_field) in enumerate(score_vec_col.items()):
            # Create a grid for the quiver plot
            vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
            anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
            vec_magn = np.sqrt((vec_proj ** 2).sum(axis=-1))
            # axs[i].invert_yaxis()
            # show the magnitude of the vector field
            im = axs[i].imshow(vec_magn.reshape(grid_nums), origin="lower", 
                          extent=[plane_coords[:, 1].min(), plane_coords[:, 1].max(), 
                                  plane_coords[:, 0].min(), plane_coords[:, 0].max(), ],
                          vmin=0, vmax=vec_magn.max())
            axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], 
                          color="white", cmap='viridis_r',
                          angles='xy')
            axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
            axs[i].invert_yaxis()
            axs[i].set_aspect('equal')
            axs[i].set_title(name, fontsize=15)
        plt.suptitle(f"score vector field {titlestr}\nsigma={sigma:.2f}", fontsize=18)
        # reduce the margin and padding to None 
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.tight_layout()
        if savefig: saveallforms(figsumdir, f"score_vector_field_{savestr}_sigma_{sigma}")
        plt.show()


def score_magnitude_projection_functional(anchors, score_func_col, score_name2plot, edm_imgshape, magnitude="l2",
                           sigmas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], titlestr="",
                           grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25),
                           figsize=(20, 5), nrowcols=(1, 4), savefig=True,
                           figsumdir=figsumdir, savestr="", show_image=True,):
    anchors_tsr = torch.stack(anchors)
    grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
            *anchors, grid_nums=grid_nums, x_range=x_range, y_range=y_range)
    if show_image:
        mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=grid_nums[1])
        plt.figure(figsize=(10, 10))
        plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if savefig: saveallforms(figsumdir, f"sample_image_map_{savestr}")
        plt.show()
    for sigma in sigmas:
        # edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
        # score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
        score_vec_col = {}
        # score_vec_col["edm NN"] = score_edm
        for score_name in score_name2plot:
            score_func = score_func_col[score_name]
            score_vec_col[score_name] = score_func(grid_vecs, sigma)
        # Calculate the vector field
        nrow, ncol = nrowcols
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
        axs = axs.flatten()
        for i, (name, vector_field) in enumerate(score_vec_col.items()):# Calculate the vector field
            # Create a grid for the quiver plot
            anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
            axs[i].invert_yaxis()
            if magnitude == "l2":
                vec_norm = torch.norm(vector_field, dim=-1).cpu().numpy()
            elif magnitude == "proj_l2":
                vec_proj = coordsys.project_vector(vector_field)
                vec_norm = torch.norm(vec_proj, dim=-1).cpu().numpy()
            elif magnitude == "ortho_l2":
                vec_ortho = coordsys.ortho_project_vector(vector_field.double())
                vec_norm = torch.norm(vec_ortho, dim=-1).cpu().numpy()
            # axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
            im = axs[i].imshow(vec_norm.reshape(grid_nums), origin="lower", 
                          extent=[plane_coords[:, 1].min(), plane_coords[:, 1].max(), 
                                  plane_coords[:, 0].min(), plane_coords[:, 0].max(), ],
                          vmin=0, vmax=vec_norm.max())
            axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
            axs[i].invert_yaxis()
            axs[i].set_aspect('equal')
            axs[i].set_title(name, fontsize=15)
            # add colorbar
            # divider = make_axes_locatable(axs[i])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, )
            # cbar.set_clim(0, vec_norm.max())
        plt.suptitle(f"score vector field {titlestr}\nsigma={sigma:2f} magnitude={magnitude}", fontsize=18)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.tight_layout()
        if savefig: saveallforms(figsumdir, f"score_magnitude_map_{savestr}_{magnitude}_sigma_{sigma}")
        plt.show()
#%%
rootdir = "../"
figroot = join(rootdir, "Figures")
figsumdir = join(figroot, "score_vector_field_vis")
os.makedirs(figsumdir, exist_ok=True)

#%% Demo model loading 
exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
# runname_long = "base_mnist_20240129-1406" # 
# epoch = 499999

runname_long = "base_mnist_20240129-1342" # base_mnist_20240129-1342
epoch = 999999

# runname_short = "base_mnist_20240130-2207"
config = get_default_config("mnist")
edm, _ = create_edm(join(exproot, runname_long, "checkpoints", f"ema_{epoch}.pth"), config, )
# %% Demo Sample generation 
seed = 42
total_steps = 18
fid_batch_size = 100
with torch.no_grad():
    noise = torch.randn([fid_batch_size, config.channels, config.img_size, config.img_size],
                        generator=torch.cuda.manual_seed(seed), device=config.device)
    samples = edm_sampler(edm, noise, num_steps=total_steps, use_ema=False).detach().cpu()
    samples.mul_(0.5).add_(0.5)
samples = torch.clamp(samples, 0., 1.).cpu()

plt.figure(figsize=(10, 10))
plt.imshow((make_grid(samples*255.0, nrow=10).permute(1, 2, 0)).numpy().astype(np.uint8))
plt.axis('off')
plt.show()

# %% [markdown]
# ### Load Dataset in EDM convention
device = "cuda"
transform = transforms.Compose([
    torchvision.transforms.Resize(32), # config.img_size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Download and load the MNIST dataset
train_edm_dataset = torchvision.datasets.MNIST(root='~/Datasets', train=True, transform=transform, download=False)
test_edm_dataset = torchvision.datasets.MNIST(root='~/Datasets', train=False, transform=transform)

edm_Xtsr = torch.stack([train_edm_dataset[i][0] for i in range(len(train_edm_dataset))])
edm_Xmat = edm_Xtsr.view(edm_Xtsr.shape[0], -1).cuda()
ytsr = torch.tensor(train_edm_dataset.targets)
edm_Xtsr_test = torch.stack([test_edm_dataset[i][0] for i in range(len(test_edm_dataset))])
edm_Xmat_test = edm_Xtsr_test.view(edm_Xtsr_test.shape[0], -1).cuda()
ytsr_test = torch.tensor(test_edm_dataset.targets)
edm_imgshape = tuple(edm_Xtsr.shape[1:])
edm_Xmean = edm_Xmat.mean(dim=0)
edm_Xcov = torch.cov(edm_Xmat.T, )
edm_std_mean = (torch.trace(edm_Xcov) / edm_Xcov.shape[0]).sqrt()

eigvals, eigvecs = torch.linalg.eigh(edm_Xcov)
eigvals = torch.flip(eigvals, dims=(0,))
eigvecs = torch.flip(eigvecs, dims=(1,))
print(eigvals.shape, eigvecs.shape)
print(eigvals[0:10].sqrt())
test_project_numerical(edm_Xmat)
# %% [markdown]
# ## Visualize learned scores by projection
# %%
# Compute score on the grid points
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        edm_Xmat[0, :], edm_Xmat[1, :], edm_Xmat[2, :], 
        grid_nums=(16, 16), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))
sigma = 0.2
edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
assert torch.allclose(plane_coords.cuda(), coordsys.project_points(grid_vecs))
plt.scatter(plane_coords[:, 0], plane_coords[:, 1])
plt.show()

#%% Compute GMM clusters

n_clusters_list = [1, 2, 5, 10, 20, 50, 100, 500] # 50, 100, 200, 500, 1000
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

#%%
# TODO: Functional dictionary for score visualization
print("Defining score functionals")
score_func_col = {
    "mean isotropic": lambda Xt, sigma: mean_isotropic_score(Xt, edm_Xmean, sigma).cpu(), 
    "mean + std isotropic": lambda Xt, sigma: mean_isotropic_score(Xt, edm_Xmean, sigma, sigma0=edm_std_mean).cpu(), 
    "gaussian": lambda Xt, sigma: Gaussian_score(Xt, edm_Xmean, edm_Xcov, sigma).cpu(), 
    "gaussian regularize": lambda Xt, sigma: Gaussian_score(Xt, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma).cpu(), 
    "gmm delta": lambda Xt, sigma: delta_GMM_score(Xt, edm_Xmat, sigma).cpu(), 
}
# Note the lambda function is tricky. 
# The default value of the function argument is evaluated at the time of the function definition, 
# not at the time of the function call. DO Not remove the nc argument in the lambda function. or all will be the same.
for n_clusters in n_clusters_list:
    score_func_col[f"gmm_{n_clusters}_mode"] = lambda Xt, sigma, nc=n_clusters: \
        gaussian_mixture_score_batch_sigma_torch(Xt, mus_col[nc].cuda(),
                Us_col[nc].cuda(), Lambdas_col[nc].cuda()+ sigma**2, weights_col[nc].cuda(), ).cpu()
#%%
def edm_score_func(Xt, sigma, edm):
    edm_Dt = edm(Xt.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(Xt.shape) - Xt) / (sigma**2)
    return score_edm
#%%
def sample_X_with_same_y(ytsr, y=None, size=(1, 3)):
    if y is None:
        y = torch.randint(0, 10, (1,)).item()
    # return RANDOM row idx in edm_Xmat with label y with size
    idx = torch.where(ytsr == y)[0]
    idx = idx[torch.randint(0, len(idx), size)]
    return idx

def sample_X_with_diff_y(ytsr, ys=None, size=3):
    if ys is None:
        # sample three different labels without replacement
        # y = torch.randint(0, 10, (1, 3))
        ys = torch.randperm(10)[:size]
    # return RANDOM row idx in edm_Xmat with label y with size
    idxs = []
    for y in ys:
        idx = torch.where(ytsr == y)[0]
        idx = idx[torch.randint(0, len(idx), (1,))]
        idxs.append(idx)
    idxs = torch.cat(idxs, dim=0)[None,]
    return idxs
#%% [markdown]
# ## Final Visualization Export 
epoch = 999999##300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
#%
sigmas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
plot_kwargs = dict(nrowcols=(2, 5), 
                   figsize=(20, 7.5),
                   show_image=True, 
                   savefig=True,
                   figsumdir=figsumdir, )
score_name2plot = ["edm NN", "gmm delta", "gmm_100_mode", "gmm_50_mode", "gmm_20_mode", "gmm_10_mode", "gmm_5_mode", "gmm_2_mode", "gaussian", "mean isotropic", ]
score_func_col["edm NN"] = lambda Xt, sigma, model=edm: edm_score_func(Xt, sigma, model).cpu()

#%% [markdown] 
# ### Different class training set samples 
# knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
# knnidx = torch.tensor([[41890, 35786, 23163]])
knnidx = sample_X_with_diff_y(ytsr, size=3)
knnidx = torch.tensor([[17960, 10172,   762]])
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu().numpy()} labels {ytsr[knnidx[0]].cpu().numpy()}"
savestr = f"ep{round(epoch/1000)}k_MNIST_train_3rnd_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr,**plot_kwargs)

#%% [markdown] 
# ### Same class training set samples 
knnidx = sample_X_with_same_y(ytsr, y=5, size=(1, 3))
knnidx = torch.tensor([[57814, 27510, 32870]])
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 samples with same class in TRAIN set, {knnidx[0].cpu().numpy()} labels {ytsr[knnidx[0]].cpu().numpy()}"
savestr = f"ep{round(epoch/1000)}k_MNIST_train_3samecls_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr,**plot_kwargs)

#%% [markdown]
# ### Different class test set samples 
# knnidx = torch.randint(0, edm_Xmat_test.shape[0], (1, 3))
knnidx = sample_X_with_diff_y(ytsr_test, size=3)
knnidx = torch.tensor([[5937, 7460, 7785]])
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TEST set, {knnidx[0].cpu().numpy()} labels {ytsr_test[knnidx[0]].cpu().numpy()}"
savestr = f"ep{round(epoch/1000)}k_MNIST_test_3rnd_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr, **plot_kwargs)

#%% [markdown]
# ### Same class test set samples 
knnidx = sample_X_with_same_y(ytsr_test, y=2, size=(1, 3))
knnidx = torch.tensor([[3658, 4066, 8719]])
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 samples with same class in TEST set, {knnidx[0].cpu().numpy()} labels {ytsr_test[knnidx[0]].cpu().numpy()}"
savestr = f"ep{round(epoch/1000)}k_MNIST_test_3samecls_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr, **plot_kwargs)





#%% [markdown]
# ## Full visualization pipeline: learning process of scores 
#%
figsumdir_tr = '/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/DiffusionHiddenLinear/score_vector_field_vis_train_proc'
os.makedirs(figsumdir_tr, exist_ok=True)

sigmas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
plot_kwargs = dict(nrowcols=(2, 5), 
                   figsize=(20, 7.5),
                   show_image=True, 
                   savefig=True,
                   figsumdir=figsumdir_tr, )
runname_init = "base_mnist_20240130-2207"
edm_col = {}
epoch_list = [500, 1000, 2500, 5000, 10000, 15000, 20000, 24999]
for epoch in epoch_list:
    edm, _ = create_edm(join(exproot, runname_init, f"checkpoints/ema_{epoch}.pth"), config)
    edm_col[epoch] = edm
    score_func_col[f"edm NN ep{epoch}"] = lambda Xt, sigma, model=edm_col[epoch]: edm_score_func(Xt, sigma, model).cpu()

score_name2plot_progress = [f"edm NN ep{epoch}" for epoch in [500, 1000, 2500, 5000, 24999]]
score_name2plot_progress.extend([ "gaussian", "gmm_10_mode", "gmm_100_mode", "gmm_500_mode","gmm delta", ])

    
knnidx = sample_X_with_same_y(ytsr_test, y=2, size=(1, 3))
knnidx = torch.tensor([[3658, 4066, 8719]])
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 samples with same class in TEST set, {knnidx[0].cpu().numpy()} labels {ytsr_test[knnidx[0]].cpu().numpy()}"
savestr = f"init_ep{round(epoch/1000)}k_MNIST_test_3samecls_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot_progress, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr, **plot_kwargs)

#%% [markdown]
# ### Different class test set samples 
# knnidx = torch.randint(0, edm_Xmat_test.shape[0], (1, 3))
knnidx = sample_X_with_diff_y(ytsr_test, size=3)
knnidx = torch.tensor([[5937, 7460, 7785]])
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TEST set, {knnidx[0].cpu().numpy()} labels {ytsr_test[knnidx[0]].cpu().numpy()}"
savestr = f"init_ep{round(epoch/1000)}k_MNIST_test_3rnd_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot_progress, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr, **plot_kwargs)

#%%

knnidx = sample_X_with_diff_y(ytsr, size=3)
knnidx = torch.tensor([[17960, 10172,   762]])
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu().numpy()} labels {ytsr[knnidx[0]].cpu().numpy()}"
savestr = f"init_ep{round(epoch/1000)}k_MNIST_train_3rnd_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot_progress, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr,**plot_kwargs)

#%% [markdown] 
# ### Same class training set samples 
knnidx = sample_X_with_same_y(ytsr, y=5, size=(1, 3))
knnidx = torch.tensor([[57814, 27510, 32870]])
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 samples with same class in TRAIN set, {knnidx[0].cpu().numpy()} labels {ytsr[knnidx[0]].cpu().numpy()}"
savestr = f"init_ep{round(epoch/1000)}k_MNIST_train_3samecls_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
# sigmas = [0.05, 0.5, 5, 20, 50]#[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, score_func_col, score_name2plot_progress, edm_imgshape,
        sigmas=sigmas, titlestr=titlestr, savestr=savestr,**plot_kwargs)






#%%


# %% [markdown]
# ## Older Full visualization pipeline
# #### Training set, 3 Random samples 
epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
savestr = f"ep{round(epoch/1000)}k_train_3rnd_%d_%d_%d" % tuple(knnidx[0].cpu().numpy())
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr, savestr=savestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)

# %% [markdown]
# #### Training set, nearest neighbors to one point
epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
dist2anchor = torch.cdist(edm_Xmat[10:11, :], edm_Xmat)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
savestr = f"ep{round(epoch/1000)}k_train_3knn_%d_%d_%d" % tuple(knnidx[0].cpu().numpy())
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr, savestr=savestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)

# %% [markdown]
# #### Test set, nearest neighbors to one point
epoch = 300000
edm_Xmat_test = edm_Xmat_test.to(device)
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
dist2anchor = torch.cdist(edm_Xmat_test[12:13, :], edm_Xmat_test)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TEST set, {knnidx[0].cpu()} labels {ytsr_test[knnidx[0].cpu()].numpy()}"
savestr = f"ep{round(epoch/1000)}k_test_3knn_%d_%d_%d" % tuple(knnidx[0].cpu().numpy())
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr, savestr=savestr,)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)

# %% [markdown]
# #### Test set: random point samples
epoch = 300000
edm_Xmat_test = edm_Xmat_test.to(device)
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
# dist2anchor = torch.cdist(edm_Xmat_test[10:11, :], edm_Xmat_test)
# knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
# knnidx = torch.randint(edm_Xmat_test.shape[0], (1, 3))
knnidx = torch.tensor([[8847, 8929, 9063]])
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TEST set, {knnidx[0].cpu()} labels {ytsr_test[knnidx[0].cpu()].numpy()}"
savestr = f"ep{round(epoch/1000)}k_test_3rnd_%d_%d_%d" % tuple(knnidx[0].cpu().numpy())
print("Score vector field", titlestr)
score_slice_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, 
                       titlestr=titlestr, savestr=savestr,)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), savestr=savestr, show_image=False)

# %% Specific Visualization of vector field. 







# %% [markdown]
# #### Off plane component of score vector

# %%
epoch = 300000
edm, _ = create_edm(join(exproot,f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="proj_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
dist2anchor = torch.cdist(edm_Xmat_test[12:13, :], edm_Xmat_test)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat_test[knnidx[0,0], :], edm_Xmat_test[knnidx[0,1], :], edm_Xmat_test[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TEST set, {knnidx[0].cpu()} labels {ytsr_test[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )

# %%
epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
dist2anchor = torch.cdist(edm_Xmat[10:11, :], edm_Xmat)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 nearest samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
print("Score vector field", titlestr)
score_magnitude_projection(anchors, edm, edm_Xmean, edm_Xcov, edm_imgshape, magnitude="ortho_l2",
                       titlestr=titlestr, grid_nums=(51, 51), )



#%%

#%% Functional visualization of GMM comparison 

epoch = 300000
edm, _ = create_edm(join(exproot, f"base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
# knnidx = torch.tensor([[41890, 35786, 23163]])
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
titlestr = f"among 3 random samples in TRAIN set, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}"
savestr = f"ep{round(epoch/1000)}k_train_3rnd_%d_%d_%d_GMM" % tuple(knnidx[0].cpu().numpy())
score_name2plot = ["gmm delta", "gmm_100_mode", "gmm_50_mode", "gmm_20_mode", "gmm_10_mode", "gmm_5_mode", "gmm_2_mode", "gaussian", "mean isotropic", ]
sigmas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
score_slice_projection_functional(anchors, edm, score_func_col, score_name2plot, edm_imgshape,
            sigmas=sigmas, titlestr=titlestr, nrowcols=(2, 5), figsize=(24, 8.7), 
            figsumdir=figsumdir, savestr=savestr, show_image=True,)
score_magnitude_projection_functional(anchors, edm, score_func_col, score_name2plot, edm_imgshape, 
            magnitude="ortho_l2",
            sigmas=sigmas, titlestr=titlestr, nrowcols=(2, 5), figsize=(24, 8.7), 
            figsumdir=figsumdir, savestr=savestr, show_image=False,)
score_magnitude_projection_functional(anchors, edm, score_func_col, score_name2plot, edm_imgshape, 
            magnitude="proj_l2",
            sigmas=sigmas, titlestr=titlestr, nrowcols=(2, 5), figsize=(24, 8.7), 
            figsumdir=figsumdir, savestr=savestr, show_image=False,)














#%% Scratch zone 

#%%
# sample three sampels from training set with the same label / class ytsr
while True:
    knnidx = torch.randint(0, edm_Xmat.shape[0], (1, 3))
    if ytsr[knnidx[0,0]] == ytsr[knnidx[0,1]] and ytsr[knnidx[0,1]] == ytsr[knnidx[0,2]]:
        break


# %% [markdown]
# ### Demo visualization without functionals 

# %%
idxs = np.random.choice(edm_Xmat.shape[0], 3, replace=False)
anchors = [edm_Xmat[idxs[0], :], edm_Xmat[idxs[1], :], edm_Xmat[idxs[2], :]]
anchors_tsr = torch.stack(anchors)
print(f"Score vector field among 3 random samples, {idxs} labels {ytsr[idxs].numpy()}")
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))

mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=25)
plt.figure(figsize=(10, 10))
plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
epoch = 300000
edm, _ = create_edm(join(exproot, "base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
for sigma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                        ("gmm delta", score_gmm_Xt),
                        ("gaussian", score_gaussian_Xt), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]):
        # Create a grid for the quiver plot
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
        axs[i].invert_yaxis()
        axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
        axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
        axs[i].set_aspect('equal')
        axs[i].set_title(name+f" score vector field\nsigma={sigma:f}")
    plt.show()


# %%
idxs = np.random.choice(edm_Xmat.shape[0], 3, replace=False)
anchors = [edm_Xmat[idxs[0], :], edm_Xmat[idxs[1], :], edm_Xmat[idxs[2], :]]
anchors_tsr = torch.stack(anchors)
print(f"Score vector field among 3 random samples, {idxs} labels {ytsr[idxs].numpy()}")
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))

mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=25)
plt.figure(figsize=(10, 10))
plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
epoch = 300000
edm, _ = create_edm(join(exproot, "base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
for sigma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 8.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                        ("gmm delta", score_gmm_Xt),
                        ("gaussian", score_gaussian_Xt), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]):
        # Create a grid for the quiver plot
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
        axs[i].invert_yaxis()
        axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
        axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
        axs[i].set_aspect('equal')
        axs[i].set_title(name+f" score vector field\nsigma={sigma:f}")
    plt.show()


# %%
import matplotlib.pyplot as plt
dist2anchor = torch.cdist(edm_Xmat[0:1, :], edm_Xmat)
knndist, knnidx = torch.topk(dist2anchor, k=3, dim=-1, largest=False)
anchors = [edm_Xmat[knnidx[0,0], :], edm_Xmat[knnidx[0,1], :], edm_Xmat[knnidx[0,2], :]]
# anchors = [edm_Xmat[idxs[0], :], edm_Xmat[idxs[1], :], edm_Xmat[idxs[2], :]]
anchors_tsr = torch.stack(anchors)
print(f"Score vector field among 3 random samples, {knnidx[0].cpu()} labels {ytsr[knnidx[0].cpu()].numpy()}")
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(25, 25), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))

mtg = make_grid(grid_vecs.reshape(-1, *edm_imgshape), nrow=25)
plt.figure(figsize=(10, 10))
plt.imshow(mtg.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
epoch = 300000
edm, _ = create_edm(join(exproot, "base_mnist_20240129-1342/checkpoints/ema_{epoch}.pth"), config)
for sigma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, (name, vector_field) in enumerate([("edm NN", score_edm),
                        ("gmm delta", score_gmm_Xt),
                        ("gaussian", score_gaussian_Xt), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]):
        # Create a grid for the quiver plot
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        anchor_proj = coordsys.project_points(anchors_tsr).cpu().numpy()
        axs[i].invert_yaxis()
        axs[i].quiver(plane_coords[:, 1], plane_coords[:, 0], vec_proj[:, 1], vec_proj[:, 0], angles='xy')
        axs[i].scatter(anchor_proj[:, 1], anchor_proj[:, 0], color='r', marker='x')
        axs[i].set_aspect('equal')
        axs[i].set_title(name+f" score vector field\nsigma={sigma:f}")
    plt.show()



# %% [markdown]
# ### Combined Vector field comparison

# %%
import matplotlib.pyplot as plt
anchors = [edm_Xmat[0, :], edm_Xmat[1, :], edm_Xmat[4, :]]
grid_vecs, norm_coords, plane_coords, coordsys = orthogonal_grid_torch(
        *anchors, grid_nums=(16, 16), x_range=(-0.25, 1.25), y_range=(-0.25, 1.25))
anchors_tsr = torch.stack(anchors)

# sigma = 1.0
for sigma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 3.5, 5.0]:
    edm_Dt = edm(grid_vecs.view(-1, *edm_imgshape), torch.tensor(sigma), None, use_ema=False).detach()
    score_edm = (edm_Dt.view(grid_vecs.shape) - grid_vecs) / (sigma**2)
    score_mean_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma)
    score_mean_std_Xt = mean_isotropic_score(grid_vecs, edm_Xmean, sigma, sigma0=edm_std_mean)
    score_gaussian_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gaussian_reg_Xt = Gaussian_score(grid_vecs, edm_Xmean, edm_Xcov + torch.eye(edm_Xcov.shape[0]).to(device) * 1E-4, sigma)
    # score_gaussian1st_Xt = Gaussian_1stexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    # score_gaussian2nd_Xt = Gaussian_2ndexpens_score(grid_vecs, edm_Xmean, edm_Xcov, sigma)
    score_gmm_Xt = delta_GMM_score(grid_vecs, edm_Xmat, sigma)
    # Calculate the vector field

    fig, ax = plt.subplots()
    ax.scatter(coordsys.project_points(anchors_tsr).cpu()[:, 0],
            coordsys.project_points(anchors_tsr).cpu()[:, 1], color='r', marker='x')
    for name, vector_field, clr in [("edm", score_edm, "black"),
                        ("gmm delta", score_gmm_Xt, "red"),
                        ("gaussian", score_gaussian_Xt, "blue"), 
                        # ("gaussian regularize", score_gaussian_reg_Xt),
                        # ("mean isotropic", score_mean_Xt), 
                        # ("mean + std isotropic", score_mean_std_Xt), 
                        # ("gaussian 1st expansion", score_gaussian1st_Xt), 
                        # ("gaussian 2nd expansion", score_gaussian2nd_Xt), 
                        ]:
        vec_proj = coordsys.project_vector(vector_field).cpu().numpy()
        ax.quiver(plane_coords[:, 0], plane_coords[:, 1], vec_proj[:, 0], vec_proj[:, 1], 
                label=name, color=clr, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_title(f"Score vector field comparison\nsigma={sigma:.2f}")
    plt.legend()
    plt.show()



