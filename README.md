# IDIM - ID as a Model-Free Measure of Class Imbalance

Welcome! This is the project page for ["Intrinsic Dimensionality as a Model-Free Measure of Class Imbalance"][https://arxiv.org/abs/2511.10475]

Some parts of this repository are *currently under construction* 🚧. Stay tuned!

## Usage

This directory contains code for using our ID-based method with variety of methods:
- Bag of Tricks (Zhang et al., 2021)
- Logit Adjustment (Menon et al., 2021)
- DRO-LT (Samuel and Chechik, 2021)
- BCL (Zhu et al., 2022)
- GLMC (Du et al., 2023)
- SURE (Li et al., 2024)

We provide scripts to estimate ID on CIFAR-LT, PlacesLT and ImageNet-LT datasets in the `id-estimation` directory.
For integration with other methods, we provide further details on how to use our method under `ID.md` under the respective directories.

### Citation

If you like this work, you can cite it as:

@article{eser2025intrinsic,
  title={Intrinsic Dimensionality as a Model-Free Measure of Class Imbalance},
  author={Cagri Eser and Zeynep Sonat Baltaci and Emre Akbas and Sinan Kalkan},
  journal={arXiv preprint arXiv:2511.10475},
  year={2025}
}
