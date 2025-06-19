# Gaussian Teleportation Diffusion
---

Plug and play acceleration using analytical solution of diffusion model and image statistics. 

![](media/Figure_Schematics.png)


<details>
<summary>Mathematical Details</summary>

The Gaussian Teleportation Diffusion process leverages analytical solutions of diffusion models combined with image statistics. Here are the key mathematical components:
$$$$

<img src="media/Figure_Schematics-01_main.png" width="800"/>

</details>


## User Guide
### Code Demo 




If you are curious about the analytical diffusion trajectory of delta mixture and general Gaussian mixture models, we provided the demo: 
```python
from gaussian_teleport import demo_delta_gmm_diffusion, demo_gaussian_mixture_diffusion
fig1 = demo_delta_gmm_diffusion(nreps=500, mus=None, sigma=1E-5)
fig1.show()
fig2 = demo_gaussian_mixture_diffusion(nreps=500, mus=None, Us=None, Lambdas=None)
fig2.show()
```

### Image Quality / Speed


### Pre-computed Image Statistics
We host the pre-computed mean and covariance matrices for common image datasets. 

* MNIST
* CIFAR10
* FFHQ64
* AFHQ64
* ImagNet64

### Organization 
- `gaussian_teleport` contains the core libraries for the project, including functions to compute analytical scores and analytical diffusion trajectories. 
- `Tables` contains most of pre-computed tables, easy to reproduce figures or analysis in the paper. 


### Run benchmark experiments yourself
```python

```

### Combined with `diffusers`

