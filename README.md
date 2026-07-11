# Persian OCR Benchmarks on Google Colab

This repository is a small Persian OCR benchmark dataset and a reproducible
guide for running OCR/VLM comparisons on Google Colab.

Start here: [COLAB_GUIDE.md](COLAB_GUIDE.md)

The benchmark catalog is in [models.yaml](models.yaml), sorted by approximate
model size with Hugging Face and GitHub links.

The benchmark contains paired images and reference transcriptions in
[`small_bench/`](small_bench/):

- `typed/`: 10 printed Persian samples
- `hand-written/`: 10 handwritten or informal samples
- each image has a same-name `.md` reference

No local Python package or runner is required. The guide uses notebook cells
with standard Colab installs, model-specific inference snippets, and a small
evaluation cell based on CER, WER, and exact line match.
