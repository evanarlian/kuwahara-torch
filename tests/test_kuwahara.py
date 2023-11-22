import pytest
import torch
from kuwahara_torch.functional import kuwahara


@pytest.mark.parametrize(
    "n,c,h,w,kernel_size",
    [
        (1, 3, 100, 100, 5),
        (2, 3, 456, 567, 11),
        (1, 3, 3, 3, 3),
    ],
)
def test_kuwahara_shape(n, c, h, w, kernel_size):
    arr = torch.rand(n, c, h, w)
    result = kuwahara(arr, kernel_size)
    assert result.size() == (n, c, h - kernel_size + 1, w - kernel_size + 1)
