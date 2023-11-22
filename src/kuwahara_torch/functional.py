import torch
import torch.nn.functional as F
from torch import Tensor


def rgb_to_gray(rgb: Tensor) -> Tensor:
    """Converts RGB to grayscale

    Args:
        rgb: RGB image tensor (n, c, h, w)

    Returns:
        grayscale: grayscale image (n, c, h, w)
    """
    # https://en.wikipedia.org/wiki/Grayscale
    orig_shape = rgb.size()
    converter = torch.tensor([0.299, 0.587, 0.114]).to(rgb.device)
    return (rgb * converter[:, None, None]).sum(1, keepdim=True).expand(orig_shape)


def kuwahara(arr: Tensor, kernel_size: int) -> Tensor:
    """Run the image with standard Kuwahara filter

    Args:
        arr (Tensor): RGB image (n, c, h, w)
        kernel_size (int): Kernel size (k, k)

    Raises:
        ValueError: If the kernel_size is even or smaller than 3

    Returns:
        Tensor: RGB image after Kuwahara (n, c, h, w)
    """
    if not (kernel_size >= 3 and kernel_size % 2 == 1):
        raise ValueError("kernel_size must be odd and at least 3")

    # the channel is kept in the gray image to reduce complex shape juggling later on
    luma = rgb_to_gray(arr)  # (n, c, h, w)

    quad = kernel_size // 2 + 1  # quadrant size (q, q)
    stride = kernel_size // 2  # stride for quadrant that results in 4 quads in a kernel

    # calculate each quadrant std, we can use variance instead to skip sqrt calculation
    luma = luma.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w, kh)
    luma = luma.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w', kh, kw)
    luma = luma.unfold(dimension=-2, size=quad, step=stride)  # (n, c, h', w', kh', kw, qh)
    luma = luma.unfold(dimension=-2, size=quad, step=stride)  # (n, c, h', w', kh', kw', qh, qw)
    stds = luma.var(dim=[-1, -2]).view(*luma.size()[:4], -1)  # (n, c, h', w', 4)

    # calculate each quadrant mean
    arr = arr.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w, kh)
    arr = arr.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w', kh, kw)
    arr = arr.unfold(dimension=-2, size=quad, step=stride)  # (n, c, h', w', kh', kw, qh)
    arr = arr.unfold(dimension=-2, size=quad, step=stride)  # (n, c, h', w', kh', kw', qh, qw)
    means = arr.mean(dim=[-1, -2]).view(*arr.size()[:4], -1)  # (n, c, h', w', 4)

    # use the smallest variance to choose the mean
    stds_argmin = stds.argmin(-1, keepdim=True)  # (n, c, h', w', 1)
    means_chosen = means.gather(-1, stds_argmin).squeeze(-1)  # (n, c, h', w')
    return means_chosen
