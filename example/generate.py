from PIL import Image

from kuwahara_torch import generalized_kuwahara, kuwahara, to_pil, to_torch


def main():
    # originally kuwahara was used as denoiser
    noisy = Image.open("example/noisy.jpg").convert("RGB")
    noisy_tensor = to_torch(noisy)
    noisy_k = to_pil(kuwahara(noisy_tensor, kernel_size=5, padding_mode="reflect"))
    noisy_k.save("example/noisy_k.jpg")
    noisy_gk = to_pil(generalized_kuwahara(noisy_tensor, kernel_size=5, padding_mode="reflect"))
    noisy_gk.save("example/noisy_gk.jpg")

    # but now it is used because of artistic effect (painterly style)
    cat = Image.open("example/cat.jpg").convert("RGB")
    cat_tensor = to_torch(cat).cuda()  # can use gpu
    cat_k = to_pil(kuwahara(cat_tensor, kernel_size=9, padding_mode="reflect").cpu())
    cat_k.save("example/cat_k.jpg")
    cat_gk = to_pil(
        generalized_kuwahara(cat_tensor, kernel_size=9, q=4, padding_mode="reflect").cpu()
    )
    cat_gk.save("example/cat_gk.jpg")

    # for huge image, cpu will be super slow, and my gpu implementation is not that
    # optimized so the memory usage will be high, to reduce, use bfloat16 instead of float16
    # because numerical stability issues
    chinatown = Image.open("example/chinatown.jpg").convert("RGB")
    chinatown_tensor = to_torch(chinatown).bfloat16().cuda()
    # apply multiple times!
    chinatown_tensor = generalized_kuwahara(chinatown_tensor, kernel_size=5, padding_mode="reflect")
    chinatown_tensor = generalized_kuwahara(
        chinatown_tensor, kernel_size=15, kernel_std=5, q=6, padding_mode="reflect"
    )
    chinatown_gk = to_pil(chinatown_tensor.cpu())
    chinatown_gk.save("example/chinatown_gk.jpg")


if __name__ == "__main__":
    main()
