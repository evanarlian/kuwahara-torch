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


def kuwahara(arr: Tensor, kernel_size: int, padding_mode: str | None = None) -> Tensor:
    """Run the image with standard Kuwahara filter

    Args:
        arr (Tensor): RGB image (n, c, h, w)
        kernel_size (int): Kernel size (k, k)
        padding_mode (str): Padding mode for torch F.pad. Defaults to no padding

    Raises:
        ValueError: If the kernel_size is even or smaller than 3

    Returns:
        Tensor: RGB image after Kuwahara (n, c, h, w)
    """
    if not (kernel_size >= 3 and kernel_size % 2 == 1):
        raise ValueError("kernel_size must be odd and at least 3")

    quad = kernel_size // 2 + 1  # quadrant size (q, q)
    stride = kernel_size // 2  # stride for quadrant that results in 4 quads in a kernel
    if padding_mode is not None:
        arr = F.pad(arr, (stride, stride, stride, stride), padding_mode)

    # the channel is kept in the gray image to reduce complex shape juggling later on
    luma = rgb_to_gray(arr)  # (n, c, h, w)

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


def generate_8_slices(kernel_size: int) -> Tensor:
    """Generate 8 triangle slices for generalized Kuwahara filter.

    Args:
        kernel_size (int): The size of the filter

    Returns:
        Tensor: Long tensor of filled with 0-7 (k, k)
    """
    center = kernel_size // 2
    a = torch.linspace(-center, center, steps=kernel_size)
    yy, xx = torch.meshgrid(a, a, indexing="ij")
    rotation = torch.atan2(yy, xx)  # range (-pi, pi)
    # remap (-pi, pi) to (0, 7) while adding half-pizza rotation
    half_pizza = 1.0 / 8 / 2
    rotation = ((rotation / torch.pi / 2.0 + 0.5 + half_pizza) * 8.0).long() % 8
    return rotation


def gaussian_kernel_2d(k: int, std: float, normalize: bool = True) -> Tensor:
    """Generate Gaussian 2d filter.
    $$G_{\sigma}(x,y) = \frac{1}{2\pi\sigma^2} \text{exp}\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

    Args:
        k (int): Kernel size
        std (float): Standard deviation
        normalize (bool):
            If true (default), the sum will be 1
            IF false, the sum will be < 1, but preserve the actual gaussian
    Returns:
        Tensor: Gaussian kernel 2d (k, k)
    """
    assert k % 2 == 1
    x = torch.linspace(-(k // 2), k // 2, k)
    # calculate the "back" part of the equation, at this point they are not normalized
    gauss1d = torch.exp(-(x**2) / (2 * std**2))
    gauss2d = torch.outer(gauss1d, gauss1d)
    if normalize:
        gauss2d /= gauss2d.sum()
    else:
        gauss2d /= 2 * torch.pi * std**2
    return gauss2d


def generalized_kuwahara(arr: Tensor, kernel_size: int, padding_mode: str | None = None) -> Tensor:
    """Run the image with generalized Kuwahara filter

    Args:
        arr (Tensor): RGB image (n, c, h, w)
        kernel_size (int): Kernel size (k, k)
        padding_mode (str): Padding mode for torch F.pad. Defaults to no padding

    Raises:
        ValueError: If the kernel_size is even or smaller than 3

    Returns:
        Tensor: RGB image after Kuwahara (n, c, h, w)
    """
    if not (kernel_size >= 3 and kernel_size % 2 == 1):
        raise ValueError("kernel_size must be odd and at least 3")

    # quad = kernel_size // 2 + 1  # quadrant size (q, q)
    stride = kernel_size // 2  # stride for quadrant that results in 4 quads in a kernel
    if padding_mode is not None:
        arr = F.pad(arr, (stride, stride, stride, stride), padding_mode)

    # the channel is kept in the gray image to reduce complex shape juggling later on
    luma = rgb_to_gray(arr)  # (n, c, h, w)

    # calculate each quadrant std, we can use variance instead to skip sqrt calculation
    luma = luma.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w, kh)
    luma = luma.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w', kh, kw)
    print(luma.size())
    return
    stds = luma.var(dim=[-1, -2]).view(*luma.size()[:4], -1)  # (n, c, h', w', 4)

    # calculate each quadrant mean
    arr = arr.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w, kh)
    arr = arr.unfold(dimension=-2, size=kernel_size, step=1)  # (n, c, h', w', kh, kw)
    means = arr.mean(dim=[-1, -2]).view(*arr.size()[:4], -1)  # (n, c, h', w', 4)

    # use the smallest variance to choose the mean
    stds_argmin = stds.argmin(-1, keepdim=True)  # (n, c, h', w', 1)
    means_chosen = means.gather(-1, stds_argmin).squeeze(-1)  # (n, c, h', w')
    return means_chosen


def main():
    arr = torch.arange(1 * 3 * 13 * 17).view(1, 3, 13, 17).float()
    generalized_kuwahara(arr, kernel_size=5)


if __name__ == "__main__":
    main()
