# HeteroFed-IDS

**HeteroFed-IDS: A Personalised Asynchronous Federated Intrusion Detection Framework for Heterogeneous IoT Edge Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Paper:** HeteroFed-IDS: A Personalised Asynchronous Federated Intrusion Detection Framework for Heterogeneous IoT Edge Networks  
> **Author:** Aniket Gundecha, MIT Academy of Engineering, Pune, India  
> **Submitted to:** IEEE Access

---

## Overview

HeteroFed-IDS resolves three structural challenges that block deployment of federated intrusion detection in real-world IoT networks:

| Challenge | HeteroFed-IDS Component |
|---|---|
| Non-IID traffic across devices | **DBEC** — Device-Behavioural Embedding Clustering |
| Heterogeneous hardware (RPi5 ↔ ESP32) | **CCKD** — Cross-Client Knowledge Distillation |
| Straggler-induced delay | **SAA** — Staleness-Aware Asynchronous Aggregation |

### Key Results (CICIoT2023, 50 clients, Dirichlet α=0.1)

| Metric | HeteroFed-IDS | FedAvg | FedMADE |
|---|---|---|---|
| Macro-F1 | **98.71%** | 93.92% | 97.40% |
| Per-round uplink | **0.42 MB** | 4.82 MB | 4.82 MB |
| Convergence (rounds) | **31** | 84 | 47 |
| Energy / round (RPi 5) | **4.2 J** | 6.2 J | 5.5 J |
| Byzantine robustness (20% attack) | **96.10%** | 80.20% | 90.13% |

---

## Repository Structure

```
heterofed-ids/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/
│   └── config.yaml              # All hyperparameters (matches paper exactly)
│
├── data/
│   ├── generate_data.py         # Synthetic CICIoT2023-style dataset generator
│   ├── dataloader.py            # Dirichlet non-IID partitioner + FL DataLoader
│   └── dataset_info.md          # Links to real public datasets used in paper
│
├── models/
│   ├── cnn_bilstm.py            # Gateway model (CNN-BiLSTM, ~4.81 MB)
│   ├── mlp.py                   # Micro-controller model (3-layer MLP, ~0.38 MB)
│   └── encoder.py               # Frozen DBEC behavioural encoder
│
├── experiments/
│   ├── dbec.py                  # Algorithm 1: DBEC clustering
│   ├── saa.py                   # Algorithm 2: SAA aggregation
│   ├── cckd.py                  # Algorithm 3: CCKD distillation
│   ├── baselines.py             # FedAvg, FedProx, SCAFFOLD, FedNova, MOON, Ditto, FedMADE
│   └── heterofed_ids.py         # Algorithm 4: Full HeteroFed-IDS framework
│
├── utils/
│   ├── metrics.py               # Macro-F1, per-class recall, convergence tracking
│   ├── privacy.py               # Rényi DP accountant
│   └── byzantine.py             # Byzantine attack simulation
│
├── results/
│   ├── tables/                  # CSV result tables (all paper numbers)
│   └── figures/                 # Reproduction figures (Figures 4–7)
│
├── tests/
│   ├── test_dbec.py
│   ├── test_saa.py
│   ├── test_cckd.py
│   └── test_privacy.py
│
├── docs/
│   ├── algorithm_details.md     # Detailed algorithm documentation
│   └── reproducibility.md       # Step-by-step reproduction guide
│
└── run_experiments.py           # Main entry point — reproduces all paper results
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/heterofed-ids.git
cd heterofed-ids
pip install -r requirements.txt
```

### 2. Generate synthetic dataset

```bash
python data/generate_data.py --samples 200000 --out data/synthetic
```

### 3. Run full experiment suite

```bash
# All experiments (generates all tables + figures)
python run_experiments.py

# Specific experiment
python run_experiments.py --exp f1          # Table III / Figure 4
python run_experiments.py --exp ablation    # Table VII
python run_experiments.py --exp byzantine   # Table VI
python run_experiments.py --exp convergence # Figure 7

# Quick smoke test (3 rounds, 10 clients)
python run_experiments.py --quick
```

### 4. Run with real datasets

Download the real datasets (see `data/dataset_info.md`), place them in `data/real/`, then:

```bash
python run_experiments.py --dataset CICIoT2023 --data_dir data/real/CICIoT2023/
```

---

## Reproducing Paper Results

All results in the paper can be reproduced using the configurations in `configs/config.yaml`.

| Paper Table/Figure | Command |
|---|---|
| Table III — Overall Macro-F1 | `python run_experiments.py --exp f1` |
| Table IV — Per-class recall | `python run_experiments.py --exp recall` |
| Table V — Communication efficiency | `python run_experiments.py --exp comm` |
| Table VI — Byzantine robustness | `python run_experiments.py --exp byzantine` |
| Table VII — Ablation study | `python run_experiments.py --exp ablation` |
| Figure 4 — F1 bar chart | auto-generated with `--exp f1` |
| Figure 7 — Convergence curves | `python run_experiments.py --exp convergence` |

**Random seeds used:** 42, 123, 456, 789, 1011 (all 5 runs averaged, as reported in paper).

---

## Datasets

The paper uses three public benchmark datasets. They are **not** included in this repository due to size. Download them from their official sources:

| Dataset | URL | Features | Classes |
|---|---|---|---|
| CICIoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html | 47 flow-level | 33 attack |
| Edge-IIoTset | https://ieee-dataport.org/documents/edge-iiotset (DOI: 10.21227/mbc1-1h68) | 61 | 14 attack |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets | Net+OS+telemetry | 9 attack |

A **synthetic dataset generator** (`data/generate_data.py`) is provided that reproduces the statistical properties of CICIoT2023 (47 features, 33 classes, same class imbalance ratios) for code testing without requiring the full 47M-flow dataset.

---

## Hyperparameters

All hyperparameters exactly match the paper (see `configs/config.yaml`):

| Parameter | Value | Description |
|---|---|---|
| `num_clients` | 50 | FL clients (35 gateway + 15 MCU) |
| `dirichlet_alpha` | 0.1 | Non-IID label skew |
| `num_rounds` | 100 | Communication rounds |
| `local_epochs` | 3 | Local SGD epochs |
| `tau_max` | 10 | SAA staleness threshold |
| `alpha` (SAA) | 0.10 | Staleness decay |
| `temperature` | 3 | CCKD KD temperature |
| `lambda_kd` | 0.5 | KD loss weight |
| `dp_sigma` | 0.8 | Gaussian DP noise std dev |
| `C_clip` | 1.0 | L2 clipping norm |
| `probe_set_size` | 2048 | CCKD probe flows |
| `embed_dim` | 128 | DBEC embedding dimension |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gundecha2024heterofed,
  title     = {HeteroFed-IDS: A Personalised Asynchronous Federated Intrusion Detection 
               Framework for Heterogeneous IoT Edge Networks},
  author    = {Gundecha, Aniket},
  journal   = {IEEE Access},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.XXXXXXX}
}
```

**Software DOI (Zenodo):** https://doi.org/10.5281/zenodo.XXXXXXX

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Aniket Gundecha**  
Department of Electronic and Telecommunication  
MIT Academy of Engineering, Alandi, Pune 412105, India  
📧 author@mitaoe.ac.in
