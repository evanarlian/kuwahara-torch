import pytest
import torch

from kuwahara_torch.functional import generalized_kuwahara, kuwahara


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


@pytest.mark.parametrize(
    "tensor_type",
    [
        "torch.FloatTensor",
        "torch.BFloat16Tensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.BFloat16Tensor",
    ],
)
def test_kuwahara_gpu_and_dtypes(tensor_type):
    # tensor_type is "combination" of dtype and device
    arr = torch.rand(3, 3, 100, 120).type(tensor_type)
    result = kuwahara(arr, 7)
    assert result.dtype == arr.dtype
    assert result.device == arr.device


@pytest.mark.parametrize(
    "n, c, h, w, kernel_size, padding_mode, out_h, out_w",
    [
        (1, 3, 100, 100, 5, None, 96, 96),
        (2, 3, 456, 567, 11, None, 446, 557),
        (1, 3, 3, 3, 3, None, 1, 1),
        (2, 3, 50, 67, 7, "constant", 50, 67),
        (2, 3, 50, 67, 7, "reflect", 50, 67),
        (2, 3, 50, 67, 7, "replicate", 50, 67),
        (2, 3, 50, 67, 7, "circular", 50, 67),
    ],
)
def test_generalized_kuwahara_correct_shape(n, c, h, w, kernel_size, padding_mode, out_h, out_w):
    arr = torch.rand(n, c, h, w)
    result = generalized_kuwahara(arr, kernel_size, padding_mode=padding_mode)
    assert result.size() == (n, c, out_h, out_w)


@pytest.mark.parametrize(
    "tensor_type",
    [
        "torch.FloatTensor",
        "torch.BFloat16Tensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.BFloat16Tensor",
    ],
)
def test_generalized_kuwahara_gpu_and_dtypes(tensor_type):
    # tensor_type is "combination" of dtype and device
    arr = torch.rand(3, 3, 50, 60).type(tensor_type)
    result = generalized_kuwahara(arr, 7)
    assert result.dtype == arr.dtype
    assert result.device == arr.device
