import torch
from torch import Tensor


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
        Tensor: _description_
    """
    summation = (mask * input).sum(dim=dim, keepdim=keepdim)
    n_elems = mask.sum(dim=dim, keepdim=keepdim)
    return summation / n_elems


def masked_var(input: Tensor, mask: Tensor, dim=None, *, correction=1, keepdim=False) -> Tensor:
    # TODO super ugly, break to variables
    avg = masked_mean(input, mask, dim, keepdim)
    return (mask * (input - avg) ** 2).sum(dim, keepdim) / (mask.sum(dim, keepdim) - correction)


def main():
    # TODO cleanup
    # color = torch.tensor([1, 2, 4, 7, 11, 16, 22, 29, 37]).view(3, 3)
    # value = torch.tensor([1, 1, 2, 3, 3, 3, 3, 4, 4]).view(3, 3)
    # mask = value == 3

    # ma_mean = masked_mean(color, mask)
    # ma_var = masked_var(color, mask)
    # print(ma_mean)
    # print(ma_var)
    # print(torch.tensor([7, 11, 16, 22.0]).var())
    # i want std or variance only for: 7 11 16 22

    inp = torch.tensor([[1, 2, 3]]).float()
    mask = torch.tensor([[0, 1, 1]]).float()
    masked_mean(inp, mask, dim=0)


if __name__ == "__main__":
    main()
