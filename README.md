# kuwahara-torch
Kuwahara filter in PyTorch.

# Usage
See full code and images on [example](example/) dir.
```bash
pip install -U kuwahara-torch
```

# Examples
Originally kuwahara filter was used for denoising.

| Original                | kuwahara                | Generalized kuwahara                |
|------------------------|------------------------|------------------------|
| ![](example/noisy.jpg) | ![](example/noisy_k.jpg) | ![](example/noisy_gk.jpg) |

But now it is used for artistic style.
| Original                | kuwahara                | Generalized kuwahara                |
|------------------------|------------------------|------------------------|
| ![](example/cat.jpg) | ![](example/cat_k.jpg) | ![](example/cat_gk.jpg) |

| Original                | Generalized kuwahara                |
|------------------------|------------------------|
| ![](example/chinatown.jpg) | ![](example/chinatown_gk.jpg) |

# TODO
* [x] Kuwahara
* [x] Generalized kuwahara
* [ ] Anisotropic kuwahara. Idk how to tilt the kernel in pytorch. PRs are welcome.

# References
* https://en.wikipedia.org/wiki/Kuwahara_filter
* [Generalized Kuwahara paper](https://core.ac.uk/download/pdf/148194268.pdf)
* [Anisotropic Kuwahara paper](https://www.kyprianidis.com/p/pg2009/jkyprian-pg2009.pdf)
* [Anisotropic Kuwahara but with polynomials](https://diglib.eg.org/bitstream/handle/10.2312/LocalChapterEvents.TPCG.TPCG10.025-030/025-030.pdf)
* https://docs.blender.org/manual/en/dev/compositing/types/filter/kuwahara.html
* https://www.youtube.com/watch?v=LDhN-JK3U9g
