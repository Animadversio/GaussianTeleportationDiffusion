"""
This file contains the analytical score functions for the Gaussian and delta mixture models.
Further, we include their Taylor expansion approximations to different orders. 
In the paper, we show that the Gaussian score and the delta mixture score are equivalent at least in the first order of Taylor expansion.
"""
import torch
import torch.nn.functional as F
import numpy as np

# define score approximators 
def mean_isotropic_score(Xt, Xmean, sigma, sigma0=0.0):
    score = (Xmean[None,] - Xt) / (sigma**2 + sigma0**2)
    return score

def Gaussian_score(Xt, Xmean, Xcov, sigma):
    cov_inv = torch.inverse(Xcov + torch.eye(Xcov.shape[0], device=Xcov.device) * sigma**2)
    score = torch.matmul((Xmean[None,] - Xt), cov_inv)
    return score

# Here we includes the analytical score with different expansion orders
def Gaussian_1stexpens_score(Xt, Xmean, Xcov, sigma):
    cov_inv_1stexp = (torch.eye(Xcov.shape[0], device=Xcov.device) - Xcov / sigma**2) / sigma**2
    score = torch.matmul((Xmean[None,] - Xt), cov_inv_1stexp)
    return score


def Gaussian_2ndexpens_score(Xt, Xmean, Xcov, sigma):
    cov_inv_2ndexp = (torch.eye(Xcov.shape[0], device=Xcov.device) - Xcov / sigma**2 + Xcov @ Xcov / sigma**4) / sigma**2
    score = torch.matmul((Xmean[None,] - Xt), cov_inv_2ndexp)
    return score


def Gaussian_1stexpens_regularize_score(Xt, Xmean, Xcov, sigma, shift=None):
    # varmean = torch.trace(Xcov) / Xcov.shape[0]
    varsum = torch.trace(Xcov) / 2
    cov_inv_1stexp = (torch.eye(Xcov.shape[0], device=Xcov.device) - Xcov / (varsum + sigma**2)) / sigma**2
    score = torch.matmul((Xmean[None,] - Xt), cov_inv_1stexp)
    return score


def Gaussian_2ndexpens_regularize_score(Xt, Xmean, Xcov, sigma, shift=None):
    varsum = torch.trace(Xcov) / 2
    cov_inv_2ndexp = (torch.eye(Xcov.shape[0], device=Xcov.device) - Xcov / (varsum + sigma**2) + Xcov @ Xcov / (varsum + sigma**2)**2) / sigma**2
    score = torch.matmul((Xmean[None,] - Xt), cov_inv_2ndexp)
    return score


# Here we include the score functions for the delta mixture model, and their Taylor expansion approximations to different orders.
def delta_GMM_score(Xt, Xmat, sigma, return_weights=False):
    # get squared distance matrix
    sqdist = torch.cdist(Xt, Xmat, p=2) ** 2
    weights = F.softmax(-sqdist / (2 * sigma**2), dim=1)
    score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    if return_weights:
        return score, weights
    else:
        return score


def delta_GMM_crossterm_score(Xt, Xmat, sigma, return_weights=False):
    """Drop the sample norm term in the softmax. 
    basically assuming all sample have same distance to the mean """
    Xmean = Xmat.mean(dim=0)
    cross_terms = (Xt - Xmean) @ (Xmat - Xmean).t() / sigma ** 2
    # get squared distance matrix
    weights = F.softmax(cross_terms, dim=1)
    score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    if return_weights:
        return score, weights
    else:
        return score


def delta_GMM_crossterm_gaussequiv_score(Xt, Xmat, Xmean, Xcov, sigma, return_weights=False):
    """Drop the sample norm term in the softmax. 
    basically assuming all sample have same distance to the mean """
    precmat = torch.inverse(Xcov + torch.eye(Xcov.shape[0], device=Xcov.device) * sigma**2)
    cross_terms = (Xt - Xmean) @ precmat @ (Xmat - Xmean).t() # / sigma ** 2
    # get squared distance matrix
    weights = cross_terms / Xmat.shape[0]
    # score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    # TODO: Note this is different from the formula above! the formula above is much worse approxiamtion
    # the one below is totally and numerically equivalent to the Gaussian score
    score = (Xmean - Xt) / sigma**2 + torch.matmul(weights, Xmat - Xmean) / sigma**2
    if return_weights:
        return score, weights
    else:
        return score


def delta_GMM_crossterm_1stexpand_score(Xt, Xmat, Xmean, Xcov, sigma, return_weights=False):
    """This shall be equivalent to the Gaussian 1st expansion score"""
    cross_terms = (Xt - Xmean) @ (Xmat - Xmean).t() / sigma ** 2
    # mf_normalizer = torch.einsum("ij,Bi,Bj->B", Xcov, 
    #                              (Xt - Xmean), 
    #                              (Xt - Xmean)) / sigma ** 4
    # get squared distance matrix
    weights = (1 + cross_terms) / Xmat.shape[0]
    # score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    score = (Xmean - Xt) / sigma**2 + torch.matmul(weights, Xmat - Xmean) / sigma**2
    if return_weights:
        return score, weights
    else:
        return score


def delta_GMM_crossterm_approx_score(Xt, Xmat, Xmean, Xcov, sigma, return_weights=False):
    """Apply mean field approximation to the normalization factor, 
    keep the cross term on top."""
    cross_terms = (Xt - Xmean) @ (Xmat - Xmean).t() / sigma ** 2
    mf_normalizer = torch.einsum("ij,Bi,Bj->B", Xcov, 
                                 (Xt - Xmean), 
                                 (Xt - Xmean)) / sigma ** 4
    # get squared distance matrix
    weights = (cross_terms - mf_normalizer[:, None] / 2).exp() / Xmat.shape[0]
    # score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    score = (Xmean - Xt) / sigma**2 + torch.matmul(weights, Xmat - Xmean) / sigma**2
    if return_weights:
        return score, weights
    else:
        return score


def delta_GMM_crossterm_approx_noexp_score(Xt, Xmat, Xmean, Xcov, sigma):
    """Apply mean field approximation to the normalization factor, 
    keep the cross term on top."""
    cross_terms = (Xt - Xmean) @ (Xmat - Xmean).t() / sigma ** 2
    mf_normalizer = torch.einsum("ij,Bi,Bj->B", Xcov, 
                                 (Xt - Xmean), 
                                 (Xt - Xmean)) / sigma ** 4
    # get squared distance matrix
    weights = (cross_terms - mf_normalizer[:, None] / 2) / Xmat.shape[0]
    # score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    score = (Xmean - Xt) / sigma**2 + torch.matmul(weights, Xmat - Xmean) / sigma**2
    return score


def delta_GMM_nearest_score(Xt, Xmat, sigma):
    """Find the nearest neighbor in dataset and attract to it. So the Softmax is actually one hot."""
    # get squared distance matrix
    sqdist = torch.cdist(Xt, Xmat, p=2) ** 2
    # effective one hot weight
    weights = F.softmax(-sqdist / (2 * 0.0001 ** 2), dim=1)
    score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    # score = (Xmean - Xt) / sigma**2 + torch.matmul(weights, Xmat - Xmean) / sigma**2
    return score


def delta_GMM_rndweights_score(Xt, Xmat, sigma):
    """ Use normalized uniform random number as Softmax weights """
    # get squared distance matrix
    sqdist = torch.cdist(Xt, Xmat, p=2) ** 2
    # random weights that sum up to one.
    weights = torch.rand_like(sqdist)
    weights = weights / weights.sum(dim=1, keepdim=True)
    score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    # score = (Xmean - Xt) / sigma**2 + torch.matmul(weights, Xmat - Xmean) / sigma**2
    return score


def delta_GMM_rndsample_score(Xt, Xmat, sigma):
    """ Use normalized uniform random number as Softmax weights """
    # get squared distance matrix
    sqdist = torch.cdist(Xt, Xmat, p=2) ** 2
    # random weights that sum up to one.
    weights = torch.rand_like(sqdist)
    weights = F.softmax(weights / (2 * 0.0001 ** 2), dim=1)
    score = (torch.matmul(weights, Xmat) - Xt) / sigma**2
    return score


def explained_var(vec1, vec2):
    return 1 - (vec1 - vec2).pow(2).sum() / vec1.pow(2).sum()


def explained_var_vec(vec1, vec2):
    return 1 - (vec1 - vec2).pow(2).sum(dim=-1) / vec1.pow(2).sum(dim=-1)


def sample_Xt_batch(Xmat, batch_size, sigma=0.01):
    idx = torch.randint(Xmat.shape[0], (batch_size,))
    Xt = Xmat[idx, :]
    Xt = Xt + sigma * torch.randn_like(Xt)
    return Xt


def sample_Xtyt_batch(Xmat, ytsr, batch_size, sigma=0.01):
    idx = torch.randint(Xmat.shape[0], (batch_size,))
    Xt = Xmat[idx, :]
    Xt = Xt + sigma * torch.randn_like(Xt)
    return Xt, ytsr[idx]

