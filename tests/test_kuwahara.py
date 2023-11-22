import pytest
import torch
from kuwahara_torch.functional import kuwahara


@pytest.mark.parametrize(
    "n, c, h, w, kernel_size, padding_mode, out_h, out_w",
    [
        (1, 3, 100, 100, 5, None, 96, 96),
        (2, 3, 456, 567, 11, None, 446, 557),
        (1, 3, 3, 3, 3, None, 1, 1),
        (2, 3, 200, 300, 7, "constant", 200, 300),
        (2, 3, 200, 300, 7, "reflect", 200, 300),
        (2, 3, 200, 300, 7, "replicate", 200, 300),
        (2, 3, 200, 300, 7, "circular", 200, 300),
    ],
)
def test_kuwahara_correct_shape(n, c, h, w, kernel_size, padding_mode, out_h, out_w):
    arr = torch.rand(n, c, h, w)
    result = kuwahara(arr, kernel_size, padding_mode)
    assert result.size() == (n, c, out_h, out_w)


## TODO also check it should raise later
