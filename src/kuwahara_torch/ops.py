from torch import Tensor

# NOTE: torch has masked tensor API, but it is not stable yet and implemented differently than mine


def masked_mean(input: Tensor, mask: Tensor, dim=None, keepdim=False) -> Tensor:
    """Masked mean is similar to nanmean, but you provide the mask.

    Args:
        input (Tensor): Tensor to be summed. Can broadcast.
        mask (Tensor): Float tensor {0.0, 1.0} to select which to ignore or included. Can broadcast.
        dim: Dim will be passed to torch. NOTE!!!: when using input or mask that you intend to broadcast,
            use negative dimension to prevent confusion! Since broadcasting works from the back, it is
            natural to do the same to dim.
        keepdim (bool, optional): Keepdim will be passed to torch. Defaults to False.

    Returns:
        Tensor: Tensor sum
    """
    summation = (mask * input).sum(dim=dim, keepdim=keepdim)
    n_elems = mask.sum(dim=dim, keepdim=keepdim)
    return summation / n_elems


def masked_var(input: Tensor, mask: Tensor, dim=None, *, correction=1, keepdim=False) -> Tensor:
    """Masked variance, you provide the mask.

    Args:
        input (Tensor): Tensor to calculate variance. Can broadcast.
        mask (Tensor): Float tensor {0.0, 1.0} to select which to ignore or included. Can broadcast.
        dim: Dim will be passed to torch. NOTE!!!: when using input or mask that you intend to broadcast,
            use negative dimension to prevent confusion! Since broadcasting works from the back, it is
            natural to do the same to dim.
        correction (int, optional): _description_. Defaults to 1.
        keepdim (bool, optional): Keepdim will be passed to torch. Defaults to False.

    Returns:
        Tensor: Tensor variance
    """
    avg = masked_mean(input, mask, dim, keepdim=True)  # preserve dim only when computing mean
    sum_sq = (mask * (input - avg) ** 2).sum(dim, keepdim)
    n_elems = (mask.sum(dim, keepdim) - correction).clamp_min(0)
    return sum_sq / n_elems
