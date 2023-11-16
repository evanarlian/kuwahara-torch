import numpy as np
import torch
from PIL import Image
from torch import Tensor


def to_torch(image: Image.Image) -> Tensor:
    """Converts PIL Image to torch tensor

    Args:
        image (Image.Image): RGB PIL Image

    Returns:
        Tensor: image tensor (N=1, C=3, H, W)
    """
    arr = np.array(image)
    arr = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
    return arr


def to_pil(arr: Tensor) -> Image.Image:
    """Converts torch tensor to PIL Image

    Args:
        tensor (Tensor): image tensor (N=1, C=3, H, W)

    Returns:
        Image.Image: RGB PIL Image
    """
    arr = arr.squeeze(0).permute(1, 2, 0).byte().numpy()
    image = Image.fromarray(arr)
    return image
