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

#%%
# tabdir = r"E:\OneDrive - Harvard University\NeurIPS2023_Diffusion\Tables"
imgsize_dict = {"ffhq64": 64, "afhqv264": 64, "cifar10": 32, }
max_batch_size_dict = {"ffhq64": 64, "afhqv264": 64, "cifar10": 256, }
refstats_dict = {#"ffhq64": "ffhq-256.npz",
                "ffhq64": r"ffhq-64x64.npz",
            "afhqv264": "afhqv2-64x64.npz",
            "cifar10": "cifar10-32x32.npz"}
# refstats_dict = {"ffhq64": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-256.npz",
#             "afhqv264": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz",
#             "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"}
# refstats_url = r"ffhq-64x64.npz"


tabdir = r"D:\DL_Projects\Vision\edm_analy_sample\summary"
tabdir = r"/n/scratch3/users/b/biw905/edm_analy_sample/summary"
tabdir = r"/home/binxu/DL_Projects/edm_analy_sample/summary"
os.makedirs(tabdir, exist_ok=True)
from core.utils.montage_utils import make_grid_np
#%%
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

export_dir = croproot + '_export'
os.makedirs(export_dir, exist_ok=True)
#%%
sfx = "all"
suffixes_select = suffixes
sfx = "sel4"
suffixes_select = ['skip0_noise80',
                 'skip4_noise45',
                 'skip8_noise24',
                 'skip12_noise12',
                 'skip16_noise5',
                 'skip20_noise2']
for RNDseed in range(25):
    imgcrops = []
    for suffix in suffixes_select:
        imgcrop = plt.imread(join(croproot, suffix, f"rnd{RNDseed:06d}.png"))
        imgcrop = imgcrop[:, :, :3]
        imgcrops.append(imgcrop)
    mtg = make_grid_np(imgcrops, nrow=len(suffixes_select), padding=2)
    plt.imsave(join(export_dir, f"rnd{RNDseed:06d}_{sfx}.png"), mtg)
    # plt.imshow(mtg)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
#%%
dataset_name = "afhqv264" # "afhqv264"  # "ffhq64" # "cifar10"  #
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

export_dir = croproot + '_export'
os.makedirs(export_dir, exist_ok=True)
#%%
sfx = "all"
suffixes_select = suffixes
sfx = "sel4"
suffixes_select = ['skip0_noise80',
                 'skip4_noise45',
                 'skip8_noise24',
                 'skip12_noise12',
                 'skip16_noise5',
                 'skip20_noise2']
for RNDseed in range(25):
    imgcrops = []
    for suffix in suffixes_select:
        imgcrop = plt.imread(join(croproot, suffix, f"rnd{RNDseed:06d}.png"))
        imgcrop = imgcrop[:, :, :3]
        imgcrops.append(imgcrop)
    mtg = make_grid_np(imgcrops, nrow=len(suffixes_select), padding=2)
    plt.imsave(join(export_dir, f"rnd{RNDseed:06d}_{sfx}.png"), mtg)
    # plt.imshow(mtg)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

#%%
dataset_name = "cifar10" # "afhqv264"  # "ffhq64" # "cifar10"  #
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

export_dir = croproot + '_export'
os.makedirs(export_dir, exist_ok=True)
#%%
sfx = "all"
suffixes_select = suffixes
sfx = "sel2"
suffixes_select = ['skip0_noise80',
                 'skip2_noise41',
                 'skip4_noise19',
                 'skip6_noise8',
                 'skip8_noise3',
                 'skip10_noise1',
                 'skip12_noise0']
for RNDseed in range(50):
    imgcrops = []
    for suffix in suffixes_select:
        imgcrop = plt.imread(join(croproot, suffix, f"rnd{RNDseed:06d}.png"))
        imgcrop = imgcrop[:, :, :3]
        imgcrops.append(imgcrop)
    mtg = make_grid_np(imgcrops, nrow=len(suffixes_select), padding=2)
    plt.imsave(join(export_dir, f"rnd{RNDseed:06d}_{sfx}.png"), mtg)
    # plt.imshow(mtg)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
