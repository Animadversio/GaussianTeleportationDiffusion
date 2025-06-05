import os.path
import numpy as np
import torch
from os.path import join
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from core.utils.montage_utils import make_grid_T
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def to_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    return PILimg


def show_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    PILimg.show()
    return PILimg


def save_imgrid(img_tsr, path, *args, **kwargs):
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    PILimg.save(path)
    return PILimg


def save_imgrid_by_row(img_tsr, path, n_row=5, *args, **kwargs):
    """Seperate img_tsr into rows and save them into different png files, with numbering 0-n."""
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    n_total = img_tsr.shape[0]
    row_num = np.ceil(n_total // n_row).astype(int)  # n_total // n_row + 1
    stem, ext = os.path.splitext(path)
    for i in range(row_num):
        PILimg = ToPILImage()(make_grid(img_tsr[i * n_row: (i + 1) * n_row].cpu(), n_row=5, *args, **kwargs))
        PILimg.save(stem + "_" + str(i) + ext)
    return


def saveallforms(figdirs, fignm, figh=None, fmts=("png", "pdf")):
    """Save all forms of a figure in an array of directories."""
    if type(figdirs) is str:
        figdirs = [figdirs]
    if figh is None:
        figh = plt.gcf()
    for figdir in figdirs:
        for sfx in fmts:
            figh.savefig(join(figdir, fignm+"."+sfx), bbox_inches='tight')


def showimg(ax, imgarr, cbar=False, ylabel=None):
    """Show an image in a given axis."""
    pcm = ax.imshow(imgarr)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if cbar:
        plt.colorbar(pcm, ax=ax)
    return pcm


def off_axes(axs):
    for ax in axs:
        ax.axis("off")

