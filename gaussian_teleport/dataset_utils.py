"""
This file contains the functions for downloading the EDM model and PCA files for different datasets.
"""
import os
import urllib.request

def download_edm_model(dataset_name, ckpt_dir):
    ckpt_url = {"cifar32": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
                "ffhq64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl",
                "afhq64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl"}[dataset_name]
    ckpt_filename = os.path.join(ckpt_dir, os.path.basename(ckpt_url))
    if not os.path.exists(ckpt_filename):
        urllib.request.urlretrieve(ckpt_url, ckpt_filename)
        print(f"Downloaded EDM model file {ckpt_filename}")
    else:
        print(f"EDM model file {ckpt_filename} already exists, skipping download")
    return ckpt_filename
    
    
def download_dataset_pca(dataset_name, PCA_dir):
    PCA_url = {"cifar32": "https://huggingface.co/datasets/binxu/image_datasets_PCAs/resolve/main/cifar32_PCA.pt",
                "ffhq64": "https://huggingface.co/datasets/binxu/image_datasets_PCAs/resolve/main/ffhq64_PCA.pt",
                "afhq64": "https://huggingface.co/datasets/binxu/image_datasets_PCAs/resolve/main/afhqv264_PCA.pt",
                "imagenet64": "https://huggingface.co/datasets/binxu/image_datasets_PCAs/resolve/main/imagenet64_PCA.pt"}[dataset_name]
    PCA_filename = os.path.join(PCA_dir, os.path.basename(PCA_url))
    if not os.path.exists(PCA_filename):
        try:
            from huggingface_hub import hf_hub_download
            hf_file_path = hf_hub_download(repo_id="binxu/image_datasets_PCAs", repo_type="dataset",
                               filename=os.path.basename(PCA_url), local_dir=PCA_dir)
        except:
            print(f"Failed to use huggingface_hub, using urllib to download PCA file {PCA_filename}")
            urllib.request.urlretrieve(PCA_url, PCA_filename)
        print(f"Downloaded PCA file {PCA_filename}")
    else:
        print(f"PCA file {PCA_filename} already exists, skipping download")
    return PCA_filename


def download_dataset_pca_all(ckpt_dir, PCA_dir, dataset_names=("cifar32", "ffhq64", "afhq64", "imagenet64")):
    for dataset_name in dataset_names:
        download_dataset_pca(dataset_name, PCA_dir)
        download_edm_model(dataset_name, ckpt_dir)
    return ckpt_dir, PCA_dir


if __name__ == "__main__":
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("PCAs", exist_ok=True)
    download_dataset_pca_all("ckpts", "PCAs")
