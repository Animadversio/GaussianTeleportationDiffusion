"""
This file contains the code for the general case of GMM diffusion, where the GMM is a mixture of Gaussian distributions with different means, covariances, and weights.
It contains functions to compute the density, log probability, and scores of the GMM, and the analytical solution for the reverse diffusion process.
The main function demo the reverse diffusion process of a general GMM.
"""
import numpy as np
import math
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from scipy.special import logsumexp, softmax
from scipy.integrate import solve_ivp


def _random_orthogonal_matrix(n):
    # generate random orthonormal matrix of 2d
    Q, R = np.linalg.qr(np.random.randn(n, n))
    return Q


def gaussian_logprob_score(x, mu, U, Lambda):
    ndim = x.shape[-1]
    logdetSigma = np.sum(np.log(Lambda))
    residual = (x - mu[None, :])  # [N batch, N dim]
    rot_residual = residual @ U   # [N batch, N dim]
    MHdist = np.sum(rot_residual ** 2 / Lambda[None, :], axis=-1)  # [N batch,]
    logprob = - 0.5 * (logdetSigma + MHdist) - 0.5 * ndim * np.log(2 * np.pi)  # [N batch,]
    score_vec = - (rot_residual / Lambda[None, :]) @ U.T  # [N batch, N dim]
    return logprob, score_vec


def test_gaussian_logprob_score(ndim=2, npnts=10):
    # evaluate density of Gaussian using scipy
    x = np.random.randn(npnts, ndim)  # [N batch, N dim]
    mu = np.random.randn(ndim)
    U = _random_orthogonal_matrix(ndim)

    Lambda = np.exp(5 * np.random.rand(ndim))  # np.array([5, 1])
    logprob, score_vec = gaussian_logprob_score(x, mu, U, Lambda)
    # evaluate density of Gaussian using scipy
    cov = U @ np.diag(Lambda) @ U.T
    logprob_scipy = multivariate_normal.logpdf(x, mean=mu, cov=cov)
    assert np.allclose(logprob, logprob_scipy)
    score_vec_exact = - (x - mu[None, :]) @ np.linalg.inv(cov)
    assert np.allclose(score_vec, score_vec_exact)


def gaussian_mixture_logprob_score(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model
    :param x: [N batch, N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N dim]
    :param Lambdas: [N comp, N dim]
    :param weights: [N comp,] or None
    :return:
    """
    ndim = x.shape[-1]
    logdetSigmas = np.sum(np.log(Lambdas), axis=-1)  # [N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = np.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = np.sum(rot_residuals ** 2 / Lambdas[None, :, :], axis=-1)  # [N batch, N comp]
    if weights is not None:
        logprobs = - 0.5 * (logdetSigmas[None, :] + MHdists) + np.log(
            weights)  # - 0.5 * ndim * np.log(2 * np.pi)  # [N batch, N comp]
    else:
        logprobs = - 0.5 * (logdetSigmas[None, :] + MHdists)
    participance = softmax(logprobs, axis=-1)  # [N batch, N comp]
    compo_score_vecs = np.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas[None, :, :]),
                                 Us)  # [N batch, N comp, N dim]
    score_vecs = np.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    # logsumexp trick
    logprob = logsumexp(logprobs, axis=-1)  # [N batch,]
    logprob -= 0.5 * ndim * np.log(2 * np.pi)
    return logprob, score_vecs


def gaussian_mixture_score(x, mus, Us, Lambdas, weights=None):
    """
       Evaluate log probability and score of a Gaussian mixture model
       :param x: [N batch, N dim]
       :param mus: [N comp, N dim]
       :param Us: [N comp, N dim, N dim]
       :param Lambdas: [N comp, N dim]
       :param weights: [N comp,] or None
       :return:
    """
    ndim = x.shape[-1]
    logdetSigmas = np.sum(np.log(Lambdas), axis=-1)  # [N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = np.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = np.sum(rot_residuals ** 2 / Lambdas[None, :, :], axis=-1)  # [N batch, N comp]
    if weights is not None:
        logprobs = - 0.5 * (logdetSigmas[None, :] + MHdists) + np.log(
            weights)  # - 0.5 * ndim * np.log(2 * np.pi)  # [N batch, N comp]
    else:
        logprobs = - 0.5 * (logdetSigmas[None, :] + MHdists)
    participance = softmax(logprobs, axis=-1)  # [N batch, N comp]
    compo_score_vecs = np.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas[None, :, :]),
                                 Us)  # [N batch, N comp, N dim]
    score_vecs = np.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    return score_vecs


def gaussian_mixture_logprob_score_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch, N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N dim]
    :param Lambdas: [N comp, N dim]
    :param weights: [N comp,] or None
    :return:
    """
    ndim = x.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas), dim=-1)  # [N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = torch.sum(rot_residuals ** 2 / Lambdas[None, :, :], dim=-1)  # [N batch, N comp]
    if weights is not None:
        logprobs = (-0.5 * (logdetSigmas[None, :] + MHdists) +
                    torch.log(weights))  # - 0.5 * ndim * torch.log(2 * torch.pi)  # [N batch, N comp]
    else:
        logprobs = -0.5 * (logdetSigmas[None, :] + MHdists)
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = torch.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas[None, :, :]),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    # logsumexp trick
    logprob = torch.logsumexp(logprobs, dim=-1)  # [N batch,]
    logprob -= 0.5 * ndim * math.log(2 * torch.pi)
    return logprob, score_vecs


def gaussian_mixture_score_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch, N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N dim]
    :param Lambdas: [N comp, N dim]
    :param weights: [N comp,] or None
    :return:
    """
    ndim = x.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas), dim=-1)  # [N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = torch.sum(rot_residuals ** 2 / Lambdas[None, :, :], dim=-1)  # [N batch, N comp]
    if weights is not None:
        logprobs = (-0.5 * (logdetSigmas[None, :] + MHdists) +
                    torch.log(weights))  # - 0.5 * ndim * torch.log(2 * torch.pi)  # [N batch, N comp]
    else:
        logprobs = -0.5 * (logdetSigmas[None, :] + MHdists)
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = torch.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas[None, :, :]),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    return score_vecs


def test_gaussian_mixture_unimodal_case(ndim=3, npnts=10):
    x = np.random.randn(npnts, ndim)  # [N batch, N dim]
    mu = np.random.randn(ndim)
    U = _random_orthogonal_matrix(ndim)
    Lambda = np.exp(5 * np.random.rand(ndim))  # np.array([5, 1])
    logprob, score_vec = gaussian_logprob_score(x, mu, U, Lambda)
    logprob2, score_vec2 = gaussian_mixture_logprob_score(x, mu[None, :], U[None, :, :], Lambda[None, :])
    assert np.allclose(logprob, logprob2)
    assert np.allclose(score_vec, score_vec2)
    # evaluate density of Gaussian using scipy
    cov = U @ np.diag(Lambda) @ U.T
    logprob_scipy = multivariate_normal.logpdf(x, mean=mu, cov=cov)
    assert np.allclose(logprob, logprob_scipy)
    score_vec_exact = - (x - mu[None, :]) @ np.linalg.inv(cov)
    assert np.allclose(score_vec, score_vec_exact)


def test_gaussian_mixture_multimodal_case(ncomp=2, ndim=3, npnts=10):
    x = np.random.randn(npnts, ndim)  # [N batch, N dim]
    mus = np.random.randn(ncomp, ndim)
    Us = np.stack([_random_orthogonal_matrix(ndim) for i in range(ncomp)], axis=0)
    Lambdas = np.exp(5 * np.random.rand(ncomp, ndim))  # np.array([5, 1])
    logprob, score_vec = gaussian_mixture_logprob_score(x, mus, Us, Lambdas)
    score_vec2 = gaussian_mixture_score(x, mus, Us, Lambdas)
    # evaluate density of Gaussian using scipy
    covs = np.stack([U @ np.diag(Lambda) @ U.T for U, Lambda in zip(Us, Lambdas)], axis=0)
    prob_scipy = np.zeros_like(x[:, 0])
    score_vec_scipy = np.zeros_like(x)
    for mu, cov in zip(mus, covs):
        # logprob_scipy = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        prob_comp_scipy = multivariate_normal.pdf(x, mean=mu, cov=cov)
        score_vec_comp_scipy = - (x - mu[None, :]) @ np.linalg.inv(cov)
        prob_scipy += prob_comp_scipy
        score_vec_scipy += score_vec_comp_scipy * prob_comp_scipy[:, None]
    logprob_scipy = np.log(prob_scipy)
    score_vec_scipy = score_vec_scipy / prob_scipy[:, None]
    # test numpy version and scipy version are the same
    assert np.allclose(score_vec, score_vec_scipy)
    assert np.allclose(score_vec2, score_vec_scipy)
    assert np.allclose(logprob, logprob_scipy)
    # note both logprob and logprob_scipy are differed from the true value by a constant
    # Also note that the scipy method is not numerically stable, esp in higher dimensions


def test_gaussian_mixture_multimodal_case_stable(ncomp=2, ndim=3, npnts=10):
    x = np.random.randn(npnts, ndim)  # [N batch, N dim]
    mus = np.random.randn(ncomp, ndim)
    Us = np.stack([_random_orthogonal_matrix(ndim) for i in range(ncomp)], axis=0)
    Lambdas = np.exp(5 * np.random.rand(ncomp, ndim))  # np.array([5, 1])
    logprob, score_vec = gaussian_mixture_logprob_score(x, mus, Us, Lambdas)
    score_vec2 = gaussian_mixture_score(x, mus, Us, Lambdas)
    logprob_th, score_vec_th = gaussian_mixture_logprob_score_torch(torch.tensor(x),
                                torch.tensor(mus), torch.tensor(Us), torch.tensor(Lambdas))
    score_vec2_th = gaussian_mixture_score_torch(torch.tensor(x),
                                torch.tensor(mus), torch.tensor(Us), torch.tensor(Lambdas))
    # evaluate density of Gaussian using scipy
    covs = np.stack([U @ np.diag(Lambda) @ U.T for U, Lambda in zip(Us, Lambdas)], axis=0)
    logprob_col = []
    score_vec_col = []
    for mu, cov in zip(mus, covs):
        logprob_scipy = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        # prob_comp_scipy = multivariate_normal.pdf(x, mean=mu, cov=cov)
        score_vec_comp_scipy = - (x - mu[None, :]) @ np.linalg.inv(cov)
        logprob_col.append(logprob_scipy)
        score_vec_col.append(score_vec_comp_scipy)
    logprob_arr = np.stack(logprob_col, axis=-1)
    logprob_scipy = logsumexp(logprob_arr, axis=-1)
    score_vec_arr = np.stack(score_vec_col, axis=-1)  # [N batch, N dim, N comp]
    participance = softmax(logprob_arr, axis=-1)  # [N batch, N comp]
    score_vec_scipy = np.sum(score_vec_arr * participance[:, None, :], axis=-1)
    # test numpy version and scipy version are the same
    assert np.allclose(score_vec, score_vec_scipy)
    assert np.allclose(score_vec2, score_vec_scipy)
    assert np.allclose(logprob, logprob_scipy)
    # test torch version and numpy version are the same
    assert np.allclose(logprob, logprob_th)
    assert np.allclose(score_vec, score_vec_th)
    assert np.allclose(score_vec, score_vec2_th)
    # note both logprob and logprob_scipy are differed from the true value by a constant
    # Also note that the scipy method is not numerically stable, esp in higher dimensions


# use gmm to define a score function, and then run the VP-ode
def beta(t):
    return (0.02 * t + 0.0001 * (1 - t)) * 1000


def alpha(t):
    # return np.exp(- 1000 * (0.01 * t**2 + 0.0001 * t))
    return np.exp(- 10 * t**2 - 0.1 * t) * 0.9999


def gmm_score_t(t, x, mus, Us, Lambdas, sigma=1E-6, weights=None):
    alpha_t = alpha(t)
    Lambdas_t = Lambdas * alpha_t + 1 - alpha_t
    return gaussian_mixture_score(x[None, :], mus=alpha_t * mus, Us=Us,
                                  Lambdas=Lambdas_t, weights=weights)[0, :]


def gmm_score_t_vec(t, x, mus, Us, Lambdas, sigma=1E-6, weights=None):
    alpha_t = alpha(t)
    # Lambdas_t = Lambdas * alpha_t + 1 - alpha_t
    Lambdas_t = Lambdas * alpha_t**2 + 1 - alpha_t**2
    return gaussian_mixture_score(x.T, mus=alpha_t * mus, Us=Us,
                                  Lambdas=Lambdas_t, weights=weights).T


def f_VP_gmm(t, x, mus, Us, Lambdas, sigma=1E-6, weights=None):
    alpha_t = alpha(t)
    beta_t = beta(t)
    Lambdas_t = Lambdas * alpha_t**2 + 1 - alpha_t**2
    # sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + gaussian_mixture_score(x[None, :], mus=alpha_t * mus, Us=Us,
                                  Lambdas=Lambdas_t, weights=weights)[0, :])


def f_VP_gmm_vec(t, x, mus, Us, Lambdas, sigma=1E-6, weights=None):
    alpha_t = alpha(t)
    beta_t = beta(t)
    Lambdas_t = Lambdas * alpha_t**2 + 1 - alpha_t**2
    # sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + gaussian_mixture_score(x.T, mus=alpha_t * mus, Us=Us,
                                  Lambdas=Lambdas_t, weights=weights).T)


def f_VP_gmm_noise_vec(t, x, mus, Us, Lambdas, sigma=1E-6, noise_std=0.01, weights=None):
    alpha_t = alpha(t)
    beta_t = beta(t)
    Lambdas_t = Lambdas * alpha_t**2 + 1 - alpha_t**2
    # sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + gaussian_mixture_score(x.T, mus=alpha_t * mus, Us=Us,
          Lambdas=Lambdas_t, weights=weights).T + noise_std * np.random.randn(*x.shape))


def exact_general_gmm_reverse_diff(mus, Us, Lambdas, xT, weights=None, t_eval=None, sigma=1E-6):
    sol = solve_ivp(lambda t, x: f_VP_gmm_vec(t, x, mus=mus, Us=Us, Lambdas=Lambdas, sigma=sigma, weights=weights),
                    (1, 0), xT, method="RK45",
                    vectorized=True, t_eval=t_eval)
    return sol.y[:, -1], sol


# Example to visualize the score function
def demo_gaussian_mixture_diffusion(nreps=500, mus=None, Us=None, Lambdas=None, weights=None):
    import matplotlib.pyplot as plt
    ndim = 2
    if mus is None:
        mus = np.array([[0, 0],
                        [-1, 1],
                        [2, 2], ])  # [N comp, N dim]
    if Lambdas is None:
        Lambdas = np.array([[.8, .2],
                        [.5, .2],
                        [.2, .8], ])
    if Us is None:
        Us = np.stack([_random_orthogonal_matrix(2) for i in range(3)], axis=0)

    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    pnts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    logprob_pnts, score_pnts = gaussian_mixture_logprob_score(pnts,
                               mus=mus, Us=Us, Lambdas=Lambdas, weights=weights)

    t_eval = np.linspace(1, 0, 100)
    sol_col = []
    for i in range(nreps):
        xT = np.random.randn(2)
        x0, sol = exact_general_gmm_reverse_diff(mus, Us, Lambdas, xT,
                                     weights=weights, t_eval=t_eval)
        sol_col.append(sol)

    x0_col = [sol.y[:, -1] for sol in sol_col]
    xT_col = [sol.y[:, 0] for sol in sol_col]
    x0_col = np.stack(x0_col, axis=0)
    xT_col = np.stack(xT_col, axis=0)

    figh = plt.figure(figsize=(8, 8))
    plt.contour(xx, yy, logprob_pnts.reshape(xx.shape), 30)
    for i, sol in enumerate(sol_col):
        plt.plot(sol.y[0, :], sol.y[1, :], c="k", alpha=0.1, lw=0.75,
                 label=None if i > 0 else "trajectories")

    plt.scatter(x0_col[:, 0], x0_col[:, 1], s=40, c="b", alpha=0.3, label="final x0", marker="o")
    plt.scatter(xT_col[:, 0], xT_col[:, 1], s=40, c="k", alpha=0.1, label="initial xT", marker="x")
    plt.scatter(mus[:, 0], mus[:, 1], s=64, c="r", alpha=0.3, label="GMM centers")
    plt.legend()
    plt.axis("image")
    plt.tight_layout()
    plt.show()
    return figh


if __name__ == "__main__":
    #%%
    fig = demo_gaussian_mixture_diffusion(nreps=500, mus=None, Us=None, Lambdas=None)
    test_gaussian_logprob_score(10)
    test_gaussian_mixture_unimodal_case(ndim=10, npnts=10)
    test_gaussian_mixture_multimodal_case(ncomp=10, ndim=100, npnts=10)
    test_gaussian_mixture_multimodal_case_stable(ncomp=10, ndim=500, npnts=10)
