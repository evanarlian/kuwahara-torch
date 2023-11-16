import torch
from torch import Tensor


def rgb_to_gray(rgb: Tensor) -> Tensor:
    """Converts RGB to grayscale

    Args:
        rgb: RGB image tensor (N, C, H, W)

    Returns:
        grayscale: grayscale image (N, C, H, W)
    """
    # https://en.wikipedia.org/wiki/Grayscale
    orig_shape = rgb.size()
    converter = torch.tensor([0.299, 0.587, 0.114])
    return (rgb * converter[:, None, None]).sum(1, keepdim=True).expand(orig_shape)


def kuwahara(arr: Tensor, kernel_size: int) -> Tensor:
    """Run the image with Kuwahara filter

    Args:
        arr (Tensor): RGB image (N, C, H, W)
        kernel_size (int): Kernel size (k, k)

    Raises:
        ValueError: If the kernel_size is even or smaller than 3

    Returns:
        Tensor: RGB image after Kuwahara (N, C, H, W)
    """
    if not (kernel_size >= 3 and kernel_size % 2 == 1):
        raise ValueError("kernel_size must be odd and at least 3")
    # the C is kept to reduce complex shape juggling later on
    luma = rgb_to_gray(arr)  # (N, C, H, W)

    # quadrant calculate (qs = quadrant side)
    qs = kernel_size // 2 + 1
    edges = kernel_size // 2
    # loop for every pixel in the image, this is very slow but okay for POC
    # this will also reduce the size, no padding yet TODO
    # quadrant diagram:
    # A B
    # C D
    result = torch.zeros_like(arr)
    for i in range(edges, arr.size(-2) - edges):
        for j in range(edges, arr.size(-1) - edges):
            # fmt: off
            # quadrants for calculating mean, based on the original image
            m_a = arr[..., i-qs+1:i+1, j-qs+1:j+1]  # (N, C, qs, qs) upper left
            m_b = arr[..., i-qs+1:i+1, j:j+qs]      # (N, C, qs, qs) upper right
            m_c = arr[..., i:i+qs, j-qs+1:j+1]      # (N, C, qs, qs) lower left
            m_d = arr[..., i:i+qs, j:j+qs]          # (N, C, qs, qs) lower right
            # quadrants for calculating std, based on brightness
            q_a = luma[..., i-qs+1:i+1, j-qs+1:j+1]  # (N, C, qs, qs) upper left
            q_b = luma[..., i-qs+1:i+1, j:j+qs]      # (N, C, qs, qs) upper right
            q_c = luma[..., i:i+qs, j-qs+1:j+1]      # (N, C, qs, qs) lower left
            q_d = luma[..., i:i+qs, j:j+qs]          # (N, C, qs, qs) lower right
            # fmt: on
            means = torch.stack([m_a, m_b, m_c, m_d], dim=-1)  # (N, C, qs, qs, 4)
            means = means.mean(dim=[2, 3])  # (N, C, 4)
            stds = torch.stack([q_a, q_b, q_c, q_d], dim=-1)  # (N, C, qs, qs, 4)
            stds = stds.std(dim=[2, 3])  # (N, C, 4)
            stds = stds.argmin(-1).unsqueeze(-1)  # (N, C, 1)
            result[..., i, j] = means.gather(-1, stds).squeeze(-1)  # (N, C)
    return result
