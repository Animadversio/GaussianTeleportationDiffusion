"""
Provide
- classes to compute Gaussian Mixture Model (GMM), log probability and its score. in numpy and torch.
- functions to fit the GMM to a dataset using K-means.(approximation)
- trainable DNN inspired by the GMM score structure, initialized with the GMM parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.distributions import MultivariateNormal


class GaussianMixture:
    def __init__(self, mus, covs, weights):
        """
        mus: a list of K 1d np arrays (D,)
        covs: a list of K 2d np arrays (D, D)
        weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each branch.
          They will be normalized to sum to 1. If they sum to zero, it will err.
        """
        self.n_component = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [np.linalg.inv(cov) for cov in covs]
        self.weights = np.array(weights)
        self.norm_weights = self.weights / self.weights.sum()
        self.RVs = []
        for i in range(len(mus)):
            self.RVs.append(multivariate_normal(mus[i], covs[i]))
        self.dim = len(mus[0])

    def add_component(self, mu, cov, weight=1):
        self.mus.append(mu)
        self.covs.append(cov)
        self.precs.append(np.linalg.inv(cov))
        self.RVs.append(multivariate_normal(mu, cov))
        self.weights.append(weight)
        self.norm_weights = self.weights / self.weights.sum()
        self.n_component += 1

    def pdf(self, x):
        """
          probability density (PDF) at $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        prob = np.dot(component_pdf, self.norm_weights)
        return prob

    def pdf_decompose(self, x):
        """
          probability density (PDF) at $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        prob = np.dot(component_pdf, self.norm_weights)
        return prob, component_pdf

    def score(self, x):
        """
        Compute the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        scores = np.zeros_like(x)
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            scores += participance[:, i:i + 1] * gradvec

        return scores

    def score_decompose(self, x):
        """
        Compute the grad to each branch for the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        gradvec_list = []
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            gradvec_list.append(gradvec)
            # scores += participance[:, i:i+1] * gradvec

        return gradvec_list, participance

    def sample(self, N):
        """ Draw N samples from Gaussian mixture
        Procedure:
          Draw N samples from each Gaussian
          Draw N indices, according to the weights.
          Choose sample between the branches according to the indices.
        """
        rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
        all_samples = np.array([rv.rvs(N) for rv in self.RVs])
        gmm_samps = all_samples[rand_component, np.arange(N), :]
        return gmm_samps, rand_component, all_samples

    def score_grid(self, XXYYs):
        XX = XXYYs[0]
        pnts = np.stack([YY.flatten() for YY in XXYYs], axis=1)
        score_vecs = self.score(pnts)
        prob = self.pdf(pnts)
        logprob = np.log(prob)
        prob = prob.reshape(XX.shape)
        logprob = logprob.reshape(XX.shape)
        score_vecs = score_vecs.reshape((*XX.shape, -1))
        return prob, logprob, score_vecs


class GaussianMixture_torch:
    def __init__(self, mus, covs, weights):
        """
        mus: a list of K 1d np arrays (D,)
        covs: a list of K 2d np arrays (D, D)
        weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each branch.
          They will be normalized to sum to 1. If they sum to zero, it will err.
        """
        self.n_component = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [torch.linalg.inv(cov) for cov in covs]
        if weights is None:
            self.weights = torch.ones(self.n_component)
        else:
            self.weights = torch.tensor(weights)
        self.norm_weights = self.weights / self.weights.sum()
        self.RVs = []
        for i in range(len(mus)):
            self.RVs.append(MultivariateNormal(mus[i], covs[i]))
        self.dim = len(mus[0])

    def add_component(self, mu, cov, weight=1):
        self.mus.append(mu)
        self.covs.append(cov)
        self.precs.append(torch.linalg.inv(cov))
        self.RVs.append(MultivariateNormal(mu, cov))
        self.weights.append(weight)
        self.norm_weights = self.weights / self.weights.sum()
        self.n_component += 1


    def pdf(self, x):
        """
          probability density (PDF) at $x$.
        """
        component_pdf = torch.stack([rv.log_prob(x) for rv in self.RVs]).exp().T
        prob = torch.dot(component_pdf, self.norm_weights)
        return prob

    def score(self, x):
        """
        Compute the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_logpdf = torch.stack([rv.log_prob(x) for rv in self.RVs]).T
        component_pdf_norm = torch.softmax(component_logpdf, dim=1)
        weighted_compon_pdf = component_pdf_norm * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        scores = torch.zeros_like(x)
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            scores += participance[:, i:i + 1] * gradvec

        return scores

    def score_decompose(self, x):
        """
        Compute the grad to each branch for the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_logpdf = torch.stack([rv.log_prob(x) for rv in self.RVs]).T
        component_pdf_norm = torch.softmax(component_logpdf, dim=1)
        weighted_compon_pdf = component_pdf_norm * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        gradvec_list = []
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            gradvec_list.append(gradvec)
            # scores += participance[:, i:i+1] * gradvec

        return gradvec_list, participance

    def sample(self, N):
        """ Draw N samples from Gaussian mixture
        Procedure:
          Draw N samples from each Gaussian
          Draw N indices, according to the weights.
          Choose sample between the branches according to the indices.
        """
        rand_component = torch.multinomial(self.norm_weights, N, replacement=True)
        # rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
        all_samples = torch.stack([rv.sample((N,)) for rv in self.RVs])
        gmm_samps = all_samples[rand_component, torch.arange(N), :]
        return gmm_samps, rand_component, all_samples


def quiver_plot(pnts, vecs, *args, **kwargs):
    plt.quiver(pnts[:, 0], pnts[:, 1], vecs[:, 0], vecs[:, 1], *args, **kwargs)


def marginal_prob_std(t, sigma):
  """Note that this std -> 0, when t->0
  So it's not numerically stable to sample t=0 in the dataset
  Note an earlier version missed the sqrt...
  """
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) ) # sqrt fixed Jun.19


def marginal_prob_std_np(t, sigma):
  return np.sqrt( (sigma**(2*t) - 1) / 2 / np.log(sigma) )


def diffuse_gmm(gmm, t, sigma):
  lambda_t = marginal_prob_std_np(t, sigma)**2 # variance
  noise_cov = np.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture(gmm.mus, covs_dif, gmm.weights)


def diffuse_gmm_torch(gmm, t, sigma):
  lambda_t = marginal_prob_std(t, sigma)**2 # variance
  noise_cov = torch.eye(gmm.dim) * lambda_t
  covs_dif = [cov + noise_cov for cov in gmm.covs]
  return GaussianMixture_torch(gmm.mus, covs_dif, gmm.weights)


def gaussian_mixture_score_batch_sigma_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch,N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N dim]
    :param Lambdas: [N batch, N comp, N dim]
    :param weights: [N comp,] or None
    :return:
    """
    if Lambdas.ndim == 2:
        Lambdas = Lambdas[None, :, :]
    ndim = x.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas), dim=-1)  # [N batch, N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists = torch.sum(rot_residuals ** 2 / Lambdas, dim=-1)  # [N batch, N comp]
    logprobs = -0.5 * (logdetSigmas + MHdists)  # [N batch, N comp]
    if weights is not None:
        logprobs += torch.log(weights)  # - 0.5 * ndim * torch.log(2 * torch.pi)  # [N batch, N comp]
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = torch.einsum("BCD,CED->BCE", - (rot_residuals / Lambdas),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs)  # [N batch, N dim]
    return score_vecs


def gaussian_mixture_lowrank_score_batch_sigma_torch(x,
                 mus, Us, Lambdas, sigma, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch
    :param x: [N batch,N dim]
    :param mus: [N comp, N dim]
    :param Us: [N comp, N dim, N rank]
    :param Lambdas: [N comp, N rank]
    :param sigma: [N batch,] or []
    :param weights: [N comp,] or None
    :return:
    """
    if Lambdas.ndim == 2:
        Lambdas = Lambdas[None, :, :]
    ndim = x.shape[-1]
    nrank = Us.shape[-1]
    logdetSigmas = torch.sum(torch.log(Lambdas + sigma[:, None, None] ** 2), dim=-1)  # [N batch, N comp,]
    logdetSigmas += (ndim - nrank) * 2 * torch.log(sigma)[:, None]  # [N batch, N comp,]
    residuals = (x[:, None, :] - mus[None, :, :])  # [N batch, N comp, N dim]
    residual_sqnorm = torch.sum(residuals ** 2, dim=-1)  # [N batch, N comp]
    Lambda_tilde = Lambdas / (Lambdas + sigma[:, None, None] ** 2)  # [N batch, N comp, N rank]
    rot_residuals = torch.einsum("BCD,CDE->BCE", residuals, Us)  # [N batch, N comp, N dim]
    MHdists_lowrk = torch.sum(rot_residuals ** 2 * Lambda_tilde, dim=-1)  # [N batch, N comp]
    logprobs = -0.5 * (logdetSigmas +
                       (residual_sqnorm - MHdists_lowrk) / sigma[:, None] ** 2)  # [N batch, N comp]
    if weights is not None:
        logprobs += torch.log(weights)
    participance = F.softmax(logprobs, dim=-1)  # [N batch, N comp]
    compo_score_vecs = - residuals + torch.einsum("BCD,CED->BCE",
                                    (rot_residuals * Lambda_tilde),
                                    Us)  # [N batch, N comp, N dim]
    score_vecs = torch.einsum("BC,BCE->BE", participance, compo_score_vecs) / (sigma[:, None] ** 2)  # [N batch, N dim]
    return score_vecs


from tqdm import trange, tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
# threadpoolctl is dependency of sklearn 
def compute_cluster(Xtrain_norm, n_clusters,
                       kmeans_batch=2048, 
                       kmeans_random_seed=0,
                       kmeans_verbose=0,
                       lambda_EPS=1E-5):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
            random_state=kmeans_random_seed, batch_size=kmeans_batch, verbose=kmeans_verbose)
    kmeans.fit(Xtrain_norm)
    print("Kmeans fitting completing, loss ", kmeans.inertia_)
    # covmats = []
    eigval_col = []
    eigvec_col = []
    freq_col = []
    for i in trange(kmeans.n_clusters):
        n_samples = np.sum(kmeans.labels_ == i)
        # print(i, "number of samples", n_samples)
        freq_col.append(n_samples)
        if n_samples == 1:
            covmat = torch.eye(covmat.shape[0]) * lambda_EPS
        else:
            covmat = torch.tensor(np.cov(Xtrain_norm[kmeans.labels_ == i].T))
            # PCA to reduce dimension
            covmat = covmat + lambda_EPS * torch.eye(covmat.shape[0])
        try:
            eigval, eigvec = torch.linalg.eigh(covmat.to(torch.float64).cuda())
        except:
            print(f"Singular matrix perform eigh on cpu. {n_samples} samples")
            eigval, eigvec = torch.linalg.eigh(covmat.to(torch.float64))
            eigval = eigval.to("cuda")
            eigvec = eigvec.to("cuda")
            
        eigval = eigval.flip(dims=(0,))  # sort from largest to smallest
        eigvec = eigvec.flip(dims=(1,))  # sort from largest to smallest
        eigval_col.append(eigval.cpu())
        eigvec_col.append(eigvec.cpu())
        # covmats.append(covmat)
    eigval_mat = torch.stack(eigval_col, dim=0)
    eigvec_mat = torch.stack(eigvec_col, dim=0)
    freq_vec = torch.tensor(freq_col)
    center_mat = torch.from_numpy(kmeans.cluster_centers_)
    print("cov PCA completed for each cluster.")
    return kmeans, eigval_mat, eigvec_mat, freq_vec, center_mat


def initialize_gmm_ansatz(eigval_mat, eigvec_mat, freq_vec, center_mat,
                          sigma_max=10, gmm_random_seed=42, n_rank=None, 
                          lambda_EPS=1E-5):
    ndim = center_mat.shape[1]
    n_clusters = center_mat.shape[0]
    torch.manual_seed(gmm_random_seed)
    if n_rank is None:
        gmm_km_cov = GMM_ansatz_net(ndim=ndim,
                    n_components=n_clusters, sigma=sigma_max)
        # Data PC initialization
        gmm_km_cov.logLambdas.data = torch.log(eigval_mat + lambda_EPS).float() #
        gmm_km_cov.Us.data = eigvec_mat.float()
        gmm_km_cov.logweights.data = torch.log(freq_vec / freq_vec.sum())
        gmm_km_cov.mus.data = center_mat.float()
        print("GMM ansatz model initialized.")
    else:
        print("Low rank GMM ansatz")
        assert type(n_rank) == int and n_rank > 0 and n_rank < ndim
        gmm_km_cov = GMM_ansatz_net_lowrank(ndim=ndim,
                    n_components=n_clusters, n_rank=n_rank, sigma=sigma_max)
        
        gmm_km_cov.logLambdas.data = torch.log(eigval_mat + lambda_EPS).float()[:, :n_rank]
        gmm_km_cov.Us.data = eigvec_mat.float()[:, :, :n_rank]
        gmm_km_cov.logweights.data = torch.log(freq_vec / freq_vec.sum())
        gmm_km_cov.mus.data = center_mat.float()
        print("GMM low rank ansatz model initialized.")
    return gmm_km_cov


def kmeans_initialized_gmm(Xtrain, n_clusters, 
                           n_rank=None,
                           kmeans_batch=2048, 
                           kmeans_random_seed=0,
                           kmeans_verbose=0,
                           gmm_random_seed=42, 
                           lambda_EPS=1E-5, 
                           sigma_max=10):
    kmeans, eigval_mat, eigvec_mat, freq_vec, center_mat = compute_cluster(Xtrain, n_clusters,
                          kmeans_batch=kmeans_batch, 
                          kmeans_random_seed=kmeans_random_seed,
                          kmeans_verbose=kmeans_verbose,
                          lambda_EPS=lambda_EPS)
    gmm_km_cov = initialize_gmm_ansatz(eigval_mat, eigvec_mat, freq_vec, center_mat,
                            sigma_max=sigma_max, gmm_random_seed=gmm_random_seed, n_rank=n_rank, 
                            lambda_EPS=lambda_EPS)
    return gmm_km_cov, kmeans

# %% A few trainable DNN inspired by the GMM score structure
class GMM_ansatz_net(nn.Module):

    def __init__(self, ndim, n_components, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        self.n_components = n_components
        # normalize the weights
        mus = torch.randn(n_components, ndim)
        Us = torch.randn(n_components, ndim, ndim)
        mus = mus / torch.norm(mus, dim=-1, keepdim=True)
        Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
        # TODO: orthonormalize Us
        self.mus = nn.Parameter(mus)
        self.Us = nn.Parameter(Us)
        self.logLambdas = nn.Parameter(torch.randn(n_components, ndim))
        self.logweights = nn.Parameter(torch.log(torch.ones(n_components) / n_components))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t, )
        return gaussian_mixture_score_batch_sigma_torch(x, self.mus, self.Us,
               self.logLambdas.exp()[None, :, :] + sigma[:, None, None] ** 2, self.logweights.exp())


class GMM_ansatz_net_lowrank(nn.Module):
    def __init__(self, ndim, n_components, n_rank, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        self.n_components = n_components
        # normalize the weights
        mus = torch.randn(n_components, ndim)
        Us = torch.randn(n_components, ndim, n_rank)
        mus = mus / torch.norm(mus, dim=-1, keepdim=True)
        Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
        # TODO: orthonormalize Us
        self.mus = nn.Parameter(mus)
        self.Us = nn.Parameter(Us)
        self.logLambdas = nn.Parameter(torch.randn(n_components, n_rank))
        self.logweights = nn.Parameter(torch.log(torch.ones(n_components) / n_components))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t, )
        return gaussian_mixture_lowrank_score_batch_sigma_torch(x, self.mus, self.Us,
               self.logLambdas.exp(), sigma[:], self.logweights.exp())


class Gauss_ansatz_net(nn.Module):
    def __init__(self, ndim, n_rank=None, sigma=5.0):
        super().__init__()
        self.ndim = ndim
        # normalize the weights
        mus = torch.randn(ndim)
        if n_rank is None:
            n_rank = ndim
        Us = torch.randn(ndim, n_rank)
        mus = mus / torch.norm(mus, dim=-1, keepdim=True)
        Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
        # TODO: orthonormalize Us
        self.mus = nn.Parameter(mus)
        self.Us = nn.Parameter(Us)
        self.logLambdas = nn.Parameter(torch.randn(n_rank))
        self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

    def forward(self, x, t):
        """
        x: (batch, ndim)
        sigma: (batch, )
        """
        sigma = self.marginal_prob_std_f(t, )
        # ndim = x.shape[-1]
        # nrank = Us.shape[-1]
        residuals = (x[:, :] - self.mus[None, :])  # [N batch, N dim]
        # residual_sqnorm = torch.sum(residuals ** 2, dim=-1)  # [N batch, ]
        Lambdas = self.logLambdas.exp()[None, :]
        Lambda_tilde = Lambdas / (Lambdas + sigma[:, None] ** 2)  # [N batch, N rank]
        rot_residuals = torch.einsum("BD,DE->BE", residuals, self.Us)  # [N batch, N comp, N dim]
        # MHdists_lowrk = torch.sum(rot_residuals ** 2 * Lambda_tilde, dim=-1)  # [N batch, N comp]
        compo_score_vecs = - residuals + torch.einsum("BE,DE->BD",
                              (rot_residuals * Lambda_tilde), self.Us)  # [N batch, N comp, N dim]
        score_vecs = compo_score_vecs / (sigma[:, None] ** 2)  # [N batch, N dim]
        return score_vecs


def test_lowrank_score_correct(n_components = 5, npnts = 40):
    # test low rank version
    ndim = 2
    n_rank = 1
    xs = torch.randn(npnts, ndim)
    mus = torch.randn(n_components, ndim)
    Us = torch.randn(n_components, ndim, n_rank)
    mus = mus / torch.norm(mus, dim=-1, keepdim=True)
    Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
    Us_ortho = Us[:, [-1, -2], :] * torch.tensor([1, -1])[None, :, None]
    # test ortho
    assert torch.allclose(torch.einsum("CDr,CDr->Cr", Us, Us_ortho), torch.zeros(n_components, n_rank))
    Lambdas_lowrank = torch.randn(n_components, n_rank).exp()
    # sigma = torch.tensor([1.0])
    sigma = torch.rand(npnts)
    score_lowrank = gaussian_mixture_lowrank_score_batch_sigma_torch(xs, mus, Us, Lambdas_lowrank, sigma)
    # built full rank basis
    Us_full = torch.cat((Us, Us_ortho), dim=-1)
    # build full rank noise covariance
    Lambdas_full = torch.cat((Lambdas_lowrank[None, :, :] + sigma[:, None, None] ** 2,
                              (sigma[:, None, None] ** 2).repeat(1, n_components, ndim - n_rank)), dim=-1)
    score_fullrank = gaussian_mixture_score_batch_sigma_torch(xs, mus, Us_full, Lambdas_full,)
    assert torch.allclose(score_lowrank, score_fullrank, atol=1e-4, rtol=1e-4)


def test_lowrank_gauss_score_correct(n_components = 1, npnts = 40, ndim = 3,
    n_rank = 2):
    # test low rank version
    xs = torch.randn(npnts, ndim)
    mus = torch.randn(n_components, ndim)
    Us = torch.randn(n_components, ndim, n_rank)
    ts = torch.rand(npnts)
    sigmas = marginal_prob_std(ts, 5.0)
    mus = mus / torch.norm(mus, dim=-1, keepdim=True)
    Us = Us / torch.norm(Us, dim=(-2), keepdim=True)
    # test ortho
    Lambdas_lowrank = torch.randn(n_components, n_rank).exp()
    score_lowrank = gaussian_mixture_lowrank_score_batch_sigma_torch(xs, mus, Us, Lambdas_lowrank, sigmas)

    net = Gauss_ansatz_net(ndim, n_rank, sigma=5.0)
    net.mus.data = mus[0]
    net.Us.data = Us[0]
    net.logLambdas.data = torch.log(Lambdas_lowrank)[0]
    # built full rank basis
    score_gauss = net(xs, ts)

    assert torch.allclose(score_lowrank, score_gauss, atol=1e-4, rtol=1e-4)


#%% Evaluation score functions with denoising loss. 
def denoise_loss_fn(model, x, marginal_prob_std_f, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability, sample t uniformly from [eps, 1.0]
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std_f(random_t,)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1)))
  return loss


def denoise_loss_fn_fixt(model, x, marginal_prob_std_f, t):
  """The loss function for individual time t. 
  Used for evaluating the score model at a fixed time t.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    t: Time, scalar in [eps, 1], note that t=0 is not numerically stable.
  """
  fix_t = torch.ones(x.shape[0], device=x.device) * t
  z = torch.randn_like(x)
  std = marginal_prob_std_f(fix_t,)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, fix_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1)))
  return loss



def eval_score_td(X_train_tsr, score_model_td,
                   sigma=25,
                   nepochs=20,
                   eps=1E-3,
                   batch_size=None,
                   device="cpu",):
    ndim = X_train_tsr.shape[1]
    score_model_td.to(device)
    X_train_tsr = X_train_tsr.to(device)
    marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
    pbar = trange(nepochs)
    score_model_td.eval()
    loss_traj = []
    for ep in pbar:
        if batch_size is None:
            with torch.no_grad():
                loss = denoise_loss_fn(score_model_td, X_train_tsr, marginal_prob_std_f, eps=eps)
        else:
            idx = torch.randint(0, X_train_tsr.shape[0], (batch_size,))
            with torch.no_grad():
                loss = denoise_loss_fn(score_model_td, X_train_tsr[idx], marginal_prob_std_f, eps=eps)

        pbar.set_description(f"step {ep} loss {loss.item():.3f}")
        loss_traj.append(loss.item())
    return loss_traj


def eval_score_td_fixt(X_train_tsr, score_model_td, time, 
                    sigma=25,
                    nepochs=20,
                    batch_size=None,
                    device="cpu",):
      assert time > 0 
      ndim = X_train_tsr.shape[1]
      score_model_td.to(device)
      X_train_tsr = X_train_tsr.to(device)
      marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
      pbar = trange(nepochs)
      score_model_td.eval()
      loss_traj = []
      for ep in pbar:
          if batch_size is None:
              with torch.no_grad():
                  loss = denoise_loss_fn_fixt(score_model_td, X_train_tsr, marginal_prob_std_f, time)
          else:
              idx = torch.randint(0, X_train_tsr.shape[0], (batch_size,))
              with torch.no_grad():
                  loss = denoise_loss_fn_fixt(score_model_td, X_train_tsr[idx], marginal_prob_std_f, time)
  
          pbar.set_description(f"step {ep} loss {loss.item():.3f}")
          loss_traj.append(loss.item())
      return loss_traj
  
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_score_model(Xtrain_norm, score_model, Nbatch=100, batch_size=2048, sigma_max=10, device="cuda"):
    loss_traj = eval_score_td(Xtrain_norm, 
            score_model, nepochs=Nbatch, batch_size=batch_size, 
            sigma=sigma_max, device=device) # clipnorm=1,
    # 50 full Gaussian GMM with Kmeans mean + cov initialization => 72 loss
    Nparameters = count_parameters(score_model)
    print (f"Average loss {np.mean(loss_traj):.3f}  ({batch_size}x{Nbatch} pnts)")
    print("Num of parameters", count_parameters(score_model))
    return np.mean(loss_traj), Nparameters


def eval_score_model_splittime(Xtrain_norm, score_model, time_pnts, Nbatch=100, batch_size=2048, sigma_max=10, device="cuda"):
    loss_time_arr = []
    for time in time_pnts:
        print(f"Time {time:.3f}")
        loss_traj = eval_score_td_fixt(Xtrain_norm, score_model, time, 
                nepochs=Nbatch, batch_size=batch_size, 
                sigma=sigma_max, device=device) # clipnorm=1,
        avg_loss = np.mean(loss_traj)
        loss_time_arr.append(avg_loss)
        print (f"t={time:.3f} Average loss {avg_loss:.3f}  ({batch_size}x{Nbatch} pnts)")
    # 50 full Gaussian GMM with Kmeans mean + cov initialization => 72 loss
    Nparameters = count_parameters(score_model)
    print("Num of parameters", count_parameters(score_model))
    return loss_time_arr, Nparameters
#%%
if __name__ == "__main__":
    test_lowrank_score_correct()
    test_lowrank_gauss_score_correct()