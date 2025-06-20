import numpy as np
from scipy.special import softmax, logsumexp

def GMM_density(mus, sigma, x):
    """
    :param mus: ndarray of mu, shape [Nbranch, Ndim]
    :param sigma: float, std of an isotropic Gaussian
    :param x: ndarray of x, shape [Nbatch, Ndim]
    :return: ndarray of p(x), shape [Nbatch,]
    """
    Nbranch = mus.shape[0]
    Ndim = mus.shape[1]
    sigma2 = sigma**2
    normfactor = np.sqrt((2 * np.pi * sigma)**Ndim)
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    prob = np.exp(- dist2 / sigma2 / 2, )  # [x batch, mu]
    prob_all = np.sum(prob, axis=1) / Nbranch / normfactor  # [x batch,]
    return prob_all


def GMM_logprob(mus, sigma, x):
    Nbranch = mus.shape[0]
    Ndim = mus.shape[1]
    sigma2 = sigma ** 2
    normfactor = np.sqrt((2 * np.pi * sigma) ** Ndim)
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    logprob = logsumexp(- dist2 / sigma2 / 2, axis=1)
    logprob -= np.log(Nbranch) + np.log(normfactor)
    return logprob


def GMM_scores(mus, sigma, x):
    """
    :param mus: ndarray of mu, shape [Nbranch, Ndim]
    :param sigma: float, std of an isotropic Gaussian
    :param x: ndarray of x, shape [Nbatch, Ndim]
    :return: ndarray of scores, shape [Nbatch, Ndim]
    """
    # for both input x and mus, the shape is [batch, space dim]
    sigma2 = sigma**2
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = np.sum(res ** 2, axis=-1)  # [x batch, mu]
    participance = softmax(- dist2 / sigma2 / 2, axis=1)  # [x batch, mu]
    scores = - np.einsum("ij,ijk->ik", participance, res) / sigma2   # [x batch, space dim]
    return scores


def beta(t, beta0=0.02, beta1=0.0001, nT=1000):
    # nT is the numebr of training time steps
    return (beta0 * t + beta1 * (1 - t)) * nT


def alpha(t, beta0=0.02, beta1=0.0001, nT=1000):
    # return np.exp(- 1000 * (0.01 * t**2 + 0.0001 * t))
    # return np.exp(- 10 * t**2 - 0.1 * t) * 0.9999
    return np.exp(- nT * (0.5 * (beta0 - beta1) * t**2 + beta1 * t)) * (1 - beta1)


# def alpha(t, beta0=0.02, beta1=0.0001):
#     # return np.exp(- 1000 * (0.01 * t**2 + 0.0001 * t))
#     return np.exp(- 10 * t**2 - 0.1 * t) * 0.9999
#     # return np.exp(- 1000 * (0.5 * (beta0 - beta1) * t**2 - beta1 * t)) * 0.9999


def score_t(t, x, mus, sigma=1E-6, alpha_fun=alpha):
    """Score function of p(x,t) according to VP SDE probability flow"""
    alpha_t = alpha_fun(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return GMM_scores(alpha_t * mus, sigma_t_sq, x[None, :])[0, :]


def score_t_vec(t, x, mus, sigma=1E-6, alpha_fun=alpha):
    """Vectorized version of score_t"""
    alpha_t = alpha_fun(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return GMM_scores(alpha_t * mus, sigma_t_sq, x.T).T


def f_VP(t, x, mus, sigma=1E-6, alpha_fun=alpha, beta_fun=beta):
    """Right hand side of the VP SDE probability flow ODE"""
    alpha_t = alpha_fun(t)
    beta_t = beta_fun(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, np.sqrt(sigma_t_sq), x[None, :])[0, :])


def f_VP_vec(t, x, mus, sigma=1E-6, alpha_fun=alpha, beta_fun=beta):
    """Right hand side of the VP SDE probability flow ODE, vectorized version"""
    alpha_t = alpha_fun(t)
    beta_t = beta_fun(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, np.sqrt(sigma_t_sq), x.T).T)


def f_VP_noise_vec(t, x, mus, sigma=1E-6, noise_std=0.01):
    alpha_t = alpha(t)
    beta_t = beta(t)
    sigma_t_sq = (1 - alpha_t**2) + sigma**2
    return - beta_t * (x + GMM_scores(alpha_t * mus, np.sqrt(sigma_t_sq), x.T).T + noise_std * np.random.randn(*x.shape))


from scipy.integrate import solve_ivp
def exact_delta_gmm_reverse_diff(mus, sigma, xT, t_eval=None, alpha_fun=alpha, beta_fun=beta):
    sol = solve_ivp(lambda t, x: f_VP_vec(t, x, mus, sigma=sigma, alpha_fun=alpha_fun, beta_fun=beta_fun),
                    (1, 0), xT, method="RK45",
                    vectorized=True, t_eval=t_eval)
    return sol.y[:, -1], sol


def demo_delta_gmm_diffusion(nreps=500, mus=None, sigma=1E-5):
    import matplotlib.pyplot as plt
    if mus is None:
        mus = np.array([[0, 0],
                        [1, 1],
                        [2, 2]])

    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    pnts = np.stack([xx, yy], axis=-1)
    pnts = pnts.reshape(-1, 2)
    logprob = GMM_logprob(mus, sigma, pnts)

    sol_col = []
    for i in range(nreps):
        xT = np.random.randn(2)
        x0, sol = exact_delta_gmm_reverse_diff(mus, sigma, xT, t_eval=None)
        sol_col.append(sol)

    x0_col = [sol.y[:, -1] for sol in sol_col]
    xT_col = [sol.y[:, 0] for sol in sol_col]
    x0_col = np.stack(x0_col, axis=0)
    xT_col = np.stack(xT_col, axis=0)
    figh = plt.figure(figsize=(8, 8))
    plt.contour(xx, yy, logprob.reshape(xx.shape), 50, )
    for i, sol in enumerate(sol_col):
        plt.plot(sol.y[0, :], sol.y[1, :], c="k", alpha=0.1, lw=0.75,
                 label=None if i > 0 else "trajectories")
    plt.scatter(x0_col[:, 0], x0_col[:, 1], s=40, c="b", alpha=0.3, label="final x0", marker="o")
    plt.scatter(xT_col[:, 0], xT_col[:, 1], s=40, c="k", alpha=0.1, label="initial xT", marker="x")
    plt.scatter(mus[:, 0], mus[:, 1], s=64, c="r", alpha=0.3, label="GMM centers")
    plt.axis("image")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return figh


if __name__ == "__main__":
    figh = demo_delta_gmm_diffusion(nreps=500)