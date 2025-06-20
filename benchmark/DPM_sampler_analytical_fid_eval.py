#%%
import pandas as pd
from os.path import join
from training import dataset
import os
import json
import click
import tqdm.auto as tqdm    
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from torch.utils.data import DataLoader, TensorDataset
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


def calculate_inception_stats_from_numpy(
    images_np, num_expected=None, seed=0, max_batch_size=64,
    num_workers=4, prefetch_factor=2, device=torch.device('cuda'),
):
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=False) as f:
        detector_net = pickle.load(f).to(device)

    # Prepare the dataset
    if num_expected is not None:
        images_np = images_np[:num_expected]
    # NOTE: no normalization, raw uint8 format, [0,1] will yield incorrect results!!!!
    # dataset_tensor = torch.from_numpy(images_np).float() / 255.0  # Normalize to [0, 1]
    dataset_tensor = torch.from_numpy(images_np)  # no normalization, raw uint8 format
    dataset_tensor = dataset_tensor.permute(0, 3, 1, 2)  # Ensure shape (N, 3, H, W)
    dataset_tensor = dataset_tensor.to(device)
    if dataset_tensor.shape[1] == 1:
        dataset_tensor = dataset_tensor.repeat([1, 3, 1, 1])
    elif dataset_tensor.shape[1] == 4:
        dataset_tensor = dataset_tensor[:, :3, :, :]
    assert dataset_tensor.shape[1] == 3
    assert dataset_tensor.dtype == torch.uint8
    tensor_dataset = TensorDataset(dataset_tensor)
    data_loader = DataLoader(
        tensor_dataset,
        batch_size=max_batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    # Accumulate statistics.
    print(f'Calculating statistics for {len(tensor_dataset)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, in tqdm.tqdm(data_loader, unit='batch', ):
        with torch.no_grad():
            features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
    # Calculate grand totals.
    mu /= len(tensor_dataset)
    sigma -= mu.ger(mu) * len(tensor_dataset)
    sigma /= len(tensor_dataset) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#%% Main computation loop
fid_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/fid-refs"
refdata_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
sample_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/edm_analy_sampler_benchmark/samples/"
eval_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/edm_analy_sampler_benchmark/eval/"
model_ckpt_dict = {"afhqv264": "edm-afhqv2-64x64-uncond-vp",
                   "ffhq64": "edm-ffhq-64x64-uncond-vp",
                   "cifar10": "edm-cifar10-32x32-uncond-vp"}
refdata_dict = {"afhqv264": "afhqv2-64x64.zip",
                "ffhq64": "ffhq-64x64.zip",
                "cifar10": "cifar10-32x32.zip"}
refstats_dict = {"afhqv264": "afhqv2-64x64.npz",
                "ffhq64": "ffhq-64x64.npz",
                "cifar10": "cifar10-32x32.npz"}
refstats_url_dict = {"ffhq64": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-256.npz",
            "afhqv264": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz",
            "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"}



#%%
# dataset_name = "cifar10" # "ffhq64" # "cifar10"  #
# dataset_name = "afhqv264" # "ffhq64" # "cifar10"  #
# dataset_name = "ffhq64"
@click.command()
@click.option(
    '--dataset_name',
    type=click.Choice(['cifar10', 'afhqv264', 'ffhq64']),
    default='ffhq64',
    help='Name of the dataset to evaluate.'
)
@click.option(
    '--range_start',
    type=int,
    default=None,
    help='Start index for processing subdirectories.'
)
@click.option(
    '--range_end',
    type=int,
    default=None,
    help='End index for processing subdirectories.'
)
def main(dataset_name, range_start, range_end):
    print(f"Evaluating {dataset_name}... from {range_start} to {range_end}")
    model_ckpt = model_ckpt_dict[dataset_name]
    model_dir = join(sample_root, model_ckpt)
    eval_dir = join(eval_root, model_ckpt)
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    ref = np.load(join(fid_root, refstats_dict[dataset_name]))
    ref_mu, ref_sigma = ref["mu"], ref["sigma"]
    # with dnnlib.util.open_url(refstats_url_dict[dataset_name]) as f:
    #     ref = dict(np.load(f))
    #     ref_mu, ref_sigma = ref["mu"], ref["sigma"]

    """for all subfolders in model_dir, collect all files ending with .npz, load them as numpy arrays, and concatenate them along the first axis"""
    fid_col = []
    # for subdir in ["uni_pc_bh1_25_skip20.0"]:#["dpm_solver_v3_25_skip80.0"]:#tqdm.tqdm(os.listdir(model_dir)):
    for subdir in sorted(os.listdir(model_dir))[range_start:range_end]:
        sampler_str = subdir 
        print(f"Evaluating {sampler_str}...")
        npz_files = [join(model_dir, subdir, f) for f in os.listdir(join(model_dir, subdir)) if f.endswith(".npz")]
        if len(npz_files) == 0:
            print(f"No npz files found in {model_dir}/{subdir}")
            continue
        sample_all = np.concatenate([np.load(f)["samples"] for f in npz_files], axis=0) # shape (N, 3, H, W) uint8 format
        print(sample_all.shape)
        if sample_all.shape[0] < 50000:
            print(f"Skipping {sampler_str} because it has less than 50000 samples")
            continue
        Mu, Sigma = calculate_inception_stats_from_numpy(sample_all, num_expected=50000,
                                    seed=0, max_batch_size=512, num_workers=0, prefetch_factor=None,
                                    device=torch.device('cuda'))
        # print(Mu.shape, Sigma.shape)
        fid = calculate_fid_from_inception_stats(Mu, Sigma, ref_mu, ref_sigma)
        print(f"{sampler_str} FID: {fid:.2f}")
        # save Mu, Sigma, fid to eval_dir
        np.savez(join(eval_dir, f"{sampler_str}_stats.npz"), mu=Mu, sigma=Sigma, fid=fid)
        # dump the fid number to a txt file
        json.dump({"fid": fid, "dataset": dataset_name, "ckpt": model_ckpt, "sampler": sampler_str}, open(join(eval_dir, f"{sampler_str}_fid.json"), "w")) 
        fid_col.append({"sampler": sampler_str, "fid": fid, "dataset": dataset_name, }) # "ckpt": model_ckpt, 
        # save fid_col to eval_dir
        # pd.DataFrame(fid_col).to_csv(join(eval_dir, "fid_by_sampler.csv"))
        # raise ValueError("stop here")
    pd.DataFrame(fid_col).to_csv(join(eval_dir, "fid_by_sampler.csv"))


if __name__ == "__main__":
    main()

