import pytest
import torch
from torch import inf, nan

from kuwahara_torch.ops import weighted_mean, weighted_var


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
def test_maskless_mean_vs_torch(input, dim, keepdim):
    mask = torch.ones_like(input)
    answer = weighted_mean(input, mask, dim, keepdim)
    reference = input.mean(dim, keepdim)
    assert answer.size() == reference.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(answer, reference, equal_nan=True)


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
    answer = weighted_mean(input, mask, dim, keepdim)
    assert answer.size() == result.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(answer, result, equal_nan=True)


@pytest.mark.parametrize(
    "input, dim, correction, keepdim",
    [
        (t([]), None, 777, False),  # nan
        (t([1, 3, 5]), None, 3, False),  # inf
        (t([1]), None, 0, False),
        (t([-2, 3, 8, 0, 4]), None, 0, False),
        (t([[1, 2, 3], [4, 5, 6]]), None, 0, False),
        # fmt: off
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), None, 0, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), None, 1, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -3, 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -1, 0, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 0, 1, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 2, 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -3, 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -1, 1, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 0, 2, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 2, 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 1), 1, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 2), 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 1), 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 2), 1, True),
        # fmt: on
    ],
)
def test_maskless_variance_vs_torch(input, dim, correction, keepdim):
    mask = torch.ones_like(input)
    answer = weighted_var(input, mask, dim, correction=correction, keepdim=keepdim)
    reference = input.var(dim, correction=correction, keepdim=keepdim)
    assert answer.size() == reference.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(answer, reference, equal_nan=True)


@pytest.mark.parametrize(
    "input, mask, dim, correction, keepdim, result",
    [
        (t([1, 3, 5]), t([1, 1, 1]), None, 3, False, t(inf)),
        (t([1, 3, 5]), t([1, 1, 1]), None, 0, False, t(8 / 3)),
        (t([1, 3, 5]), t([0, 0, 0]), None, 1, False, t(nan)),
        (t([1, 3, 5]), t([1, 1, 1]), 0, 2, True, t([8])),
        (t([1, 3, 5]), t([0, 0, 0]), 0, 0, True, t([nan])),
        (t([[1, 3, 5]]), t([[0, 1, 1]]), 0, 0, False, t([nan, 0, 0])),
        (t([[1, 3, 5]]), t([[0, 1, 1]]), 0, 0, True, t([[nan, 0, 0]])),
        # fmt: off
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[1, 1, 0], [0, 0, 0], [0, 0, 0]]), None, 1, False, t(0.5)),
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[0, 0, 1], [0, 0, 0], [0, 0, 0]]), None, 1, False, t(nan)),  # this is not inf because 0/0 == nan
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[0, 0, 0], [1, 1, 1], [1, 0, 0]]), None, 1, False, t(42)),
        (t([[1, 2, 4], [7, 11, 16], [22, 29, 37]]), t([[0, 0, 0], [0, 0, 0], [0, 1, 1]]), None, 1, False, t(32)),
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
            1,
            False,
            t([0.5, nan, 42, 32]),
        ),
    ],
)
def test_masked_variance(input, mask, dim, correction, keepdim, result):
    answer = weighted_var(input, mask, dim, correction=correction, keepdim=keepdim)
    assert answer.size() == result.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(answer, result, equal_nan=True)


@pytest.mark.parametrize(
    "input, dim, correction, keepdim",
    [
        (t([]), None, 777, False),  # nan
        (t([1, 3, 5]), None, 3, False),  # inf
        (t([1]), None, 0, False),
        (t([-2, 3, 8, 0, 4]), None, 0, False),
        (t([[1, 2, 3], [4, 5, 6]]), None, 0, False),
        # fmt: off
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), None, 0, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), None, 1, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -3, 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -1, 0, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 0, 1, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 2, 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -3, 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), -1, 1, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 0, 2, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), 2, 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 1), 1, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 2), 2, False),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 1), 0, True),
        (t([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]), (0, 2), 1, True),
        # fmt: on
    ],
)
def test_maskless_std_vs_torch(input, dim, correction, keepdim):
    mask = torch.ones_like(input)
    std = weighted_var(input, mask, dim, correction=correction, keepdim=keepdim).sqrt()
    reference = input.std(dim, correction=correction, keepdim=keepdim)
    assert std.size() == reference.size()  # prevents wrong shape broadcasts to true below
    assert torch.allclose(std, reference, equal_nan=True)
