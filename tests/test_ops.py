import pytest
import torch
from torch import nan

from kuwahara_torch.ops import masked_mean


def t(arr):
    """Helper function to make float torch tensor."""
    return torch.tensor(arr).float()


@pytest.mark.parametrize(
    "input, dim, keepdim",
    [
        (t([1]), None, False),
        (t([-2, 3, 8, 0, 4]), None, False),
        (t([[1, 2, 3], [4, 5, 6]]), None, False),
        # fmt: off
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), None, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), None, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -3, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -1, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 0, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -3, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -1, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 2, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 1), False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 2), False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 1), True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 2), True),
        # fmt: on
    ],
)
def test_no_mask_vs_torch(input, dim, keepdim):
    mask = torch.ones_like(input)
    answer = masked_mean(input, mask, dim, keepdim)
    reference = input.mean(dim, keepdim)
    assert answer.size() == reference.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(answer, reference)


@pytest.mark.parametrize(
    "input, mask, dim, keepdim, result",
    [
        (t([1, 2, 3]), t([1, 1, 1]), None, False, t(2)),
        (t([1, 2, 3]), t([0, 0, 0]), None, False, t(nan)),
        (t([1, 2, 3]), t([1, 1, 1]), 0, True, t([2])),
        (t([1, 2, 3]), t([0, 0, 0]), 0, True, t([nan])),
        (t([[1, 2, 3]]), t([[0, 1, 1]]), 0, False, t([nan, 2, 3])),
        (t([[1, 2, 3]]), t([[0, 1, 1]]), 0, True, t([[nan, 2, 3]])),
        # fmt: off
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[1, 1, 0], [0, 0, 0], [0, 0, 0]]), None, False, t(1.5)),
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[0, 0, 1], [0, 0, 0], [0, 0, 0]]), None, False, t(4)),
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[0, 0, 0], [1, 1, 1], [1, 0, 0]]), None, False, t(14)),
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[0, 0, 0], [0, 0, 0], [0, 1, 1]]), None, False, t(33)),
        # fmt: on
        (
            t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]),
            t(
                [
                    [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 1, 1]],
                ]
            ),
            (-1, -2),  # using negative dimensions for broadcastable items is strongly advised
            False,
            t([1.5, 4, 14, 33]),
        ),
    ],
)
def test_masked_mean(input, mask, dim, keepdim, result):
    answer = masked_mean(input, mask, dim, keepdim)
    assert answer.size() == result.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(answer, result, equal_nan=True)
