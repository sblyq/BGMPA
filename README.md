# BGMPA

This is the PyTorch implementation for **BGMPA**:
**B**ehavior-**g**uided **M**odality Graph **P**ropagation with Preference **A**ggregation for multimodal recommendation.

BGMPA addresses the mismatch between content-based modality similarity and behavior-oriented preference relevance. It preserves both plain modality graphs and behavior-guided modality graphs, performs dual-view propagation, and uses collaborative context for graph-view reliability selection and preference-guided multimodal aggregation.

## Introduction

The repository is organized as follows:

```text
BGMPA_clean_code/
  data/                     # dataset placeholder
  src/
    common/                 # trainer and base recommender
    configs/                # model, dataset, and global configs
    models/                 # BGMPA implementation
    utils/                  # data loading, evaluation, logging
    main.py                 # training entry
  requirements.txt
```

## Environment

The code has been tested with Python 3.8+ and PyTorch 1.13+.

```bash
pip install -r requirements.txt
```

## Dataset

Please place the processed Amazon datasets under `data/`:

```text
data/
  baby/
    baby.inter
    image_feat.npy
    text_feat.npy
  sports/
    sports.inter
    image_feat.npy
    text_feat.npy
  clothing/
    clothing.inter
    image_feat.npy
    text_feat.npy
```

Dataset-specific configuration files are available in `src/configs/dataset/`.

## Usage

Train BGMPA on Baby:

```bash
cd src
python main.py -m BGMPA -d baby
```

Train on Sports or Clothing:

```bash
cd src
python main.py -m BGMPA -d sports
python main.py -m BGMPA -d clothing
```

Override hyperparameters from the command line:

```bash
cd src
python main.py -m BGMPA -d baby --image_knn_k=40 --text_knn_k=10 --behavior_graph_alpha=0.6
```

The main hyperparameters are defined in `src/configs/model/BGMPA.yaml`.

## Reproducibility Notes

This clean release keeps only the core training and evaluation code. Runtime logs, cached KNN graphs, model checkpoints, processed feature files, and paper drafts are intentionally excluded from version control. Pre-computed graph files can be regenerated during training from the processed interaction and feature files.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{bgmpa2026,
  title     = {Behavior-guided Modality Graph Propagation with Preference Aggregation for Multimodal Recommendation},
  author    = {Anonymous Author(s)},
  booktitle = {Anonymous Submission},
  year      = {2026}
}
```

## Acknowledgement

The project structure follows common multimodal recommendation codebases, including MMRec-style training and evaluation organization.
