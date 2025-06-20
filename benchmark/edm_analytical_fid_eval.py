from os.path import join
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def find_unique_suffixes(folder_name):
    # List all the files in the specified folder
    file_names_list = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]

    # Regular expression pattern to match the "skipXX_noiseXX" part
    pattern = re.compile(r'skip\d+_noise\d+')

    # Extract the specific suffix pattern from each file name stem
    suffixes = [pattern.search(file_name.split('.')[0]).group() for file_name in file_names_list if
                pattern.search(file_name.split('.')[0])]

    # Get unique suffixes
    unique_suffixes = set(suffixes)

    # Define a function to extract the number after "skip"
    def get_skip_number(suffix):
        return int(suffix.split('_')[0].replace('skip', ''))

    # Sort the unique suffixes by the number after "skip"
    sorted_suffixes = sorted(unique_suffixes, key=get_skip_number)

    return sorted_suffixes


def crop_all_from_montage(img, totalnum, imgsize=32, pad=2):
    """Return all crops from a montage image"""
    nrow, ncol = (img.shape[0] - pad) // (imgsize + pad), (img.shape[1] - pad) // (imgsize + pad)
    imgcol = []
    for imgid in range(totalnum):
        ri, ci = np.unravel_index(imgid, (nrow, ncol))
        img_crop = img[pad + (pad + imgsize) * ri:pad + imgsize + (pad + imgsize) * ri, \
               pad + (pad + imgsize) * ci:pad + imgsize + (pad + imgsize) * ci, :]
        imgcol.append(img_crop)
    return imgcol


def find_files_with_suffix(folder_path, target_suffix):
    # List all files in the specified folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Filter files that contain the target suffix
    matching_files = sorted([file_name for file_name in all_files if target_suffix in file_name])

    return matching_files

#%%
import pandas as pd
from os.path import join
from training import dataset
import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
#----------------------------------------------------------------------------

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    # if dist.get_rank() != 0:
    #     torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    print(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    # if num_expected is not None and len(dataset_obj) < num_expected:
    #     raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    # if len(dataset_obj) < 2:
    #     raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    # if dist.get_rank() == 0:
    #     torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    print(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        # torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if images.shape[1] == 4:
            images = images[:, :3, :, :]
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    # torch.distributed.all_reduce(mu)
    # torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#%%
# tabdir = r"E:\OneDrive - Harvard University\NeurIPS2023_Diffusion\Tables"
tabdir = r"D:\DL_Projects\Vision\edm_analy_sample\summary"
tabdir = r"/n/scratch3/users/b/biw905/edm_analy_sample/summary"
tabdir = r"/home/binxu/DL_Projects/edm_analy_sample/summary"
imgsize_dict = {"ffhq64": 64, "afhqv264": 64, "cifar10": 32, }
max_batch_size_dict = {"ffhq64": 64, "afhqv264": 64, "cifar10": 256, }
# refstats_dict = {"ffhq64": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-256.npz",
#             "afhqv264": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz",
#             "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"}

# refstats_url = r"ffhq-64x64.npz"
refstats_dict = {#"ffhq64": "ffhq-256.npz",
                "ffhq64": r"ffhq-64x64.npz",
            "afhqv264": "afhqv2-64x64.npz",
            "cifar10": "cifar10-32x32.npz"}


dataset_name = "ffhq64" # "afhqv264"  # "ffhq64" # "cifar10"  #
figdir = rf"D:\DL_Projects\Vision\edm_analy_sample\{dataset_name}_uncond_vp_edm_theory"
figdir = rf"/n/scratch3/users/b/biw905/edm_analy_sample/{dataset_name}_uncond_vp_edm_theory"
figdir = rf"/home/binxu/DL_Projects/edm_analy_sample/{dataset_name}_uncond_vp_edm_theory"
croproot = figdir + "_crops"
imgsize = imgsize_dict[dataset_name]
max_batch_size = max_batch_size_dict[dataset_name]
refstats_url = refstats_dict[dataset_name]

suffixes = find_unique_suffixes(figdir)
for suffix in suffixes:
    os.makedirs(join(croproot, suffix), exist_ok=True)
os.makedirs(tabdir, exist_ok=True)
#%%
# load all mtg figures crop and save into folders
for suffix in suffixes:
    mtglist = find_files_with_suffix(figdir, suffix)
    for mtgname in tqdm.tqdm(mtglist):
        numbers_before_after_dash = re.findall(r'rnd(\d+)-(\d+)_', mtgname)
        rnd_start, rnd_end = numbers_before_after_dash[0]
        rnd_batch = list(range(int(rnd_start), int(rnd_end) + 1))
        mtg_arr = plt.imread(join(figdir, mtgname))
        imgcrops = crop_all_from_montage(mtg_arr, max_batch_size, imgsize=imgsize, pad=2)
        for imgcrop, rnd_id in zip(imgcrops, rnd_batch):
            plt.imsave(join(croproot, suffix, f"rnd{rnd_id:06d}.png"), imgcrop)

#%
fid_batch_size = 256
fid_col = []
for suffix in suffixes:
    Mu, Sigma = calculate_inception_stats(join(croproot, suffix), num_expected=50000,
                                   seed=0, max_batch_size=fid_batch_size, num_workers=0, prefetch_factor=None,
                                device=torch.device('cuda'))
    # https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
    with dnnlib.util.open_url(refstats_url) as f:
        ref = dict(np.load(f))

    fid = calculate_fid_from_inception_stats(Mu, Sigma, ref['mu'], ref['sigma'])
    print(f"{suffix} FID: {fid:.2f}")
    fid_col.append(fid)
#%%
# sorted(os.listdir(croproot))
# with the folder name column
df = pd.DataFrame(fid_col, columns=["FID"], index=suffixes)
# df.to_csv(join(croproot, "fid_by_skipping.csv"))
df.to_csv(join(tabdir, f"{dataset_name}_fid_by_skipping_new.csv"))





#%% DEV ZONE

# TODO: find a way to compute t_steps from skipsteps
# suffixes = find_unique_suffixes(r"D:\DL_Projects\Vision\edm_analy_sample\ffhq64_uncond_vp_edm_theory")
# # for skipstep in [0, 1, 2, 3, 4, 5, 6, 7, 8, ]:  # range(1, num_steps):
# #     sigma_max_skip = t_steps[skipstep]
# #     print(f"skip{skipstep}_noise{sigma_max_skip:.0f}")
# filenames = find_files_with_suffix(r"D:\DL_Projects\Vision\edm_analy_sample\ffhq64_uncond_vp_edm_theory",
#                                       suffixes[0])
# #%%
# figdir = "/home/binxu/DL_Projects/edm_analy_sample/cifar10_uncond_vp_edm"
# croproot = "/home/binxu/DL_Projects/edm_analy_sample/cifar10_uncond_vp_edm_crops"

#%%
# load all mtg figures crop and save into folders
seeds = list(range(50000))
max_batch_size = 256
# iterate batches
rank_batches = np.array_split(seeds, len(seeds) // max_batch_size)
for batch_seeds in rank_batches:
    for skip, sigma_max_skip in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, ],
                                    t_steps):
        assert os.path.exists(join(figdir, f"rnd{batch_seeds[0]:06d}-{batch_seeds[-1]:06d}"
                                           f"_skip{skip}_noise{sigma_max_skip:.0f}.png"))
        mtg_arr = plt.imread(join(figdir, f"rnd{batch_seeds[0]:06d}-{batch_seeds[-1]:06d}"
                                             f"_skip{skip}_noise{sigma_max_skip:.0f}.png"))
        imgcrops = crop_all_from_montage(mtg_arr, len(batch_seeds), imgsize=32, pad=2)
        for imgid, imgcrop in enumerate(imgcrops):
            plt.imsave(join(croproot, f"skip{skip}_noise{sigma_max_skip:.0f}", f"rnd{batch_seeds[imgid]:06d}.png"), imgcrop)

#%%
croproot = "/home/binxu/DL_Projects/edm_analy_sample/cifar10_uncond_vp_edm_crops"
foldername = f"skip1_noise58"
#%%
# from fid import calculate_inception_stats
# skip, sigma_max_skip = 0, t_steps[0]
fid_col = []
for foldername in sorted(os.listdir(croproot)):
    Mu, Sigma = calculate_inception_stats(join(croproot, foldername), num_expected=50000,
                                   seed=0, max_batch_size=256, num_workers=0, prefetch_factor=None,
                                device=torch.device('cuda'))

    with dnnlib.util.open_url("https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz") as f:
        ref = dict(np.load(f))

    fid = calculate_fid_from_inception_stats(Mu, Sigma, ref['mu'], ref['sigma'])
    print(f"{foldername} FID: {fid:.2f}")
    fid_col.append(fid)
#%%
# tabdir = r"E:\OneDrive - Harvard University\NeurIPS2023_Diffusion\Tables"
tabdir = r"/home/binxu/DL_Projects/edm_analy_sample/summary"

# sorted(os.listdir(croproot))
# with the folder name column
df = pd.DataFrame(fid_col, columns=["FID"], index=sorted(os.listdir(croproot)))
# df.to_csv(join(croproot, "fid_by_skipping.csv"))
df.to_csv(join(tabdir, "fid_by_skipping.csv"))


#%%
#%%