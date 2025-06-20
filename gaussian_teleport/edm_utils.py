import re
import pandas as pd
import torch
import sys
sys.path.append("../")
from easydict import EasyDict as edict

def create_edm(path, config, device="cuda"):
    from gaussian_teleport.edm.train_edm import create_model, EDM
    model = create_model(config)
    if path is not None:
        model.load_state_dict(torch.load(path))
    else:
        print("No pretrained model path is provided, use the model at initialization.")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False);
    edm = EDM(model=model, cfg=config)
    return edm, model

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2). (independent design)
## https://github.com/NVlabs/edm/blob/main/generate.py#L25
def edm_sampler(
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
    x_next = latents.to(torch.float64) * t_steps[0]
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

    return x_next


@torch.no_grad()
def edm_sampler_inpaint(
    edm, latents, target_img, mask, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    use_ema=True, fixed_noise=False
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([edm.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    initial_noise = torch.randn_like(latents)
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # x_hat = x_next
        t_hat = t_cur
        noise_perturb = initial_noise if fixed_noise else torch.randn_like(target_img)
        x_hat = (1 - mask[None, None]) * (target_img + noise_perturb * t_cur) + \
                     mask[None, None]  * x_next
        # Euler step.
        denoised = edm(x_hat, t_hat, class_labels, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, t_next, class_labels, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def parse_train_logfile(logfile_path):
    # logfile = "/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/std.log"
    # Define the regex pattern to extract the desired information
    # pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+): (.*)"
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+-->\s+step:\s+(\d+),\s+current lr:\s+([\d.]+)\s+average loss:\s+([\d.]+);\s+batch loss:\s+([\d.]+)"

    # Create empty lists to store the extracted information
    df_col = []
    # Read the logfile line by line and extract the desired information
    with open(logfile_path, "r") as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                timestamp = match.group(1)
                level = match.group(2)
                step = match.group(3)
                learning_rate = match.group(4)
                average_loss = match.group(5)
                batch_loss = match.group(6)
                df_col.append({"timestamp": timestamp, "level": level, "step": int(step),
                                "learning_rate": float(learning_rate), "average_loss": float(average_loss),
                                "batch_loss": float(batch_loss),})

    # Create a pandas dataframe from the extracted information
    df = pd.DataFrame(df_col)
    # Display the dataframe
    print(df.tail())
    return df


def get_default_config(dataset_name, **kwargs):
    if dataset_name == "mnist":
        config = {
            "DATASET": "mnist",
            "channel_mult": [1, 2, 3, 4],
            "model_channels": 16,
            "attn_resolutions": [0,],
            "layers_per_block": 1,
            "num_fid_sample": 5000,
            "fid_batch_size": 1024,
            "channels": 1,
            "img_size": 32,
            "device": "cuda",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "sigma_data": 0.5,
            "spatial_matching": "padding",
        }
    elif dataset_name == "cifar10":
        config = {
            "DATASET": "cifar10",
            "channel_mult": [1, 2, 2],
            "model_channels": 96,
            "attn_resolutions": [16,],
            "layers_per_block": 2,
            "num_fid_sample": 5000,
            "fid_batch_size": 1024,
            "channels": 3,
            "img_size": 32,
            "device": "cuda",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "sigma_data": 0.5,
            "spatial_matching": "padding",
        }
    elif dataset_name == "ffhq64":
        config = {
            "DATASET": "ffhq64",
            "channel_mult": [1, 2, 2],
            "model_channels": 96,
            "attn_resolutions": [16,],
            "layers_per_block": 2,
            "num_fid_sample": 5000,
            "fid_batch_size": 1024,
            "channels": 3,
            "img_size": 64,
            "device": "cuda",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "sigma_data": 0.5,
            "spatial_matching": "padding",
        }
    elif dataset_name == "afhq256":
        config = {
            "DATASET": "afhq256",
            "channel_mult": [1, 2, 2],
            "model_channels": 96,
            "attn_resolutions": [16,],
            "layers_per_block": 2,
            "num_fid_sample": 5000,
            "fid_batch_size": 1024,
            "channels": 3,
            "img_size": 64,
            "device": "cuda",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "sigma_data": 0.5,
            "spatial_matching": "padding",
        }
    elif dataset_name == "RAVEN10_abstract":
        config = {
            "DATASET": "RAVEN10_abstract",
            "channel_mult": [1, 2, 4, 4],
            "model_channels": 64,
            "attn_resolutions": [0, 1, 2],
            "layers_per_block": 1,
            "num_fid_sample": 5000,
            "fid_batch_size": 1024,
            "channels": 3,
            "img_size": 9,
            "device": "cuda",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "sigma_data": 0.5,
            "spatial_matching": "padding",
        }
    elif dataset_name == "RAVEN10_abstract_onehot":
        config = {
            "DATASET": "RAVEN10_abstract_onehot",
            "channel_mult": [1, 2, 4, 4],
            "model_channels": 64,
            "attn_resolutions": [0, 1, 2],
            "layers_per_block": 1,
            "num_fid_sample": 5000,
            "fid_batch_size": 1024,
            "channels": 3,
            "img_size": 9,
            "device": "cuda",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "sigma_data": 0.5,
            "spatial_matching": "padding",
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
        else:
            raise ValueError(f"Unsupported config key: {k}, recheck the config keys:\n {[*config.keys()]}")

    return edict(config)

