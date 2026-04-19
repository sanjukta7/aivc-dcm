# Discrete Cell Models (DCM): Discrete Diffusion for Single-Cell Gene Expression Modeling

Sanjukta Bhattacharya, Christian Gensbigler, Shaamil Karim, Jon Lees

[![Paper](https://img.shields.io/badge/MLGenX_2026-Poster-blue?style=for-the-badge)](https://openreview.net/forum?id=GPR1YXdE4U)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.02.19.705033-b31b1b.svg?style=for-the-badge)](https://www.biorxiv.org/content/10.64898/2026.02.19.705033v1)
[![PDF](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge)](https://openreview.net/pdf?id=GPR1YXdE4U)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Dataset Download**](#dataset-download) | [**Training**](#training) | [**Sampling**](#sampling) | [**Citation**](#citation)

<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2026-03-01] Accepted at **MLGenX @ ICLR 2026** 🎉
- [2026-03-01] Initial release


## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/YOUR_ORG/dcm.git
cd dcm
uv sync
source .venv/bin/activate
```

## Dataset Download

We use three benchmarks. Download each dataset and place it under `datasets/`:

| Dataset | Description | Link |
|---|---|---|
| `dentate-gyrus` | Hippocampal neurogenesis single-cell atlas | [Figshare](https://figshare.com/articles/dataset/Dentate_Gyrus_dataset/23354174?file=41110652) |
| `replogle` | Genome-scale Perturb-seq (Replogle et al. 2022) | [Figshare](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) |
| `pbmc-1m` | 1M PBMC cytokine perturbation (Parse) | [Figshare](https://figshare.com/articles/dataset/pbmc_parse/28589774?file=53372768) |

## Training

Set your dataset and output paths in `configs/perturb_seq_small.yaml` before training.

**Unconditional generation (Dentate Gyrus):**
```bash
uv run scripts/train_perturbseq.py \
    CONFIG=configs/perturb_seq_small.yaml \
    TRAIN_DATA_PATH=datasets/dentate_gyrus.h5ad
```

**Conditional generation with perturbation conditioning (Replogle):**
```bash
uv run scripts/train_perturbseq.py \
    CONFIG=configs/perturb_seq_small.yaml \
    TRAIN_DATA_PATH=datasets/replogle.h5ad \
    COND_LABELS_PT_PATH=datasets/protein_embeddings.pt
```

Checkpoints and logs will be saved to `experiments/dcm/<run_name>/`.

## Sampling

Run the following command to sample from a trained model:

```bash
uv run scripts/inference_conditional.py \
    EXPERIMENT_DIR=experiments/dcm \
    CELL_TYPE=hepg2 \
    NUM_SAMPLES_PER_PERT=1000 \
    NUM_STEPS=100
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{
bhattacharya2026discrete,
title={Discrete Diffusion for Single-Cell Gene Expression Modeling},
author={Sanjukta Bhattacharya and Christian Gensbigler and Shaamil Karim and Jon Lees},
booktitle={ICLR 2026 Workshop on Machine Learning for Genomics Explorations},
year={2026},
url={https://openreview.net/forum?id=GPR1YXdE4U}
}
```
Model weights are added [here](https://drive.google.com/drive/folders/1UYmGCN-wJ6zlYttG-cqtyTenVqNydw-N?usp=sharing)
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


## Acknowledgements
This repository builds heavily off of [score sde](https://github.com/yang-song/score_sde_pytorch), [sedd](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/tree/main), [DiT](https://github.com/facebookresearch/DiT) and [STATE](https://github.com/ArcInstitute/state), [scDLM](https://github.com/czi-ai/scLDM). We also used the [cell-load](https://github.com/ArcInstitute/cell-load) package introduced in the STATE repository.
