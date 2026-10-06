# Reproducibility Guide

This document provides step-by-step instructions to reproduce every
table and figure in the paper from scratch.

---

## Environment Setup

```bash
git clone https://github.com/YOUR_USERNAME/heterofed-ids.git
cd heterofed-ids
pip install -r requirements.txt
```

Tested on:
- Python 3.10, Ubuntu 22.04
- PyTorch 2.1 (CPU and CUDA 11.8)
- Flower 1.8

---

## Option A — Synthetic Dataset (no download required)

```bash
# Generate synthetic CICIoT2023-style dataset (200k flows, all 33 classes)
python data/generate_data.py --samples 200000 --seed 42 --out data/synthetic

# Run all experiments and generate all figures
python run_experiments.py

# Or run specific experiments
python run_experiments.py --exp f1          # Figure 4, Table III
python run_experiments.py --exp ablation    # Table VII
python run_experiments.py --exp byzantine   # Table VI
python run_experiments.py --exp convergence # Figure 7
```

---

## Option B — Real Datasets

### Step 1: Download datasets

| Dataset | URL |
|---|---|
| CICIoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| Edge-IIoTset | https://ieee-dataport.org/documents/edge-iiotset |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets |

Place downloaded files in `data/real/<DATASET_NAME>/`.

### Step 2: Preprocess

```bash
python data/preprocess.py --dataset CICIoT2023   --in data/real/CICIoT2023/   --out data/processed/
python data/preprocess.py --dataset Edge-IIoTset  --in data/real/Edge-IIoTset/  --out data/processed/
python data/preprocess.py --dataset TON_IoT       --in data/real/TON_IoT/       --out data/processed/
```

### Step 3: Run experiments

```bash
python run_experiments.py --dataset CICIoT2023 --data_dir data/processed/
```

---

## Hyperparameters

All hyperparameters are in `configs/config.yaml` and match the paper exactly.
Key values:

| Parameter | Value |
|---|---|
| Random seeds | 42, 123, 456, 789, 1011 |
| Dirichlet α | 0.1 |
| Clients | 50 (35 gateway + 15 MCU) |
| Rounds | 100 |
| τ_max (SAA) | 10 |
| σ (DP) | 0.8 |
| C_clip (DP) | 1.0 |
| KD temperature | 3 |
| λ_KD | 0.5 |

---

## Expected Outputs

After running `python run_experiments.py`, the following files are created:

```
results/
├── tables/
│   ├── table3_overall_f1.csv        ← Table III
│   ├── table4_minority_recall.csv   ← Table IV
│   ├── table5_communication.csv     ← Table V
│   ├── table6_byzantine.csv         ← Table VI
│   └── table7_ablation.csv          ← Table VII
└── figures/
    ├── figure4_f1.png               ← Figure 4
    ├── figure5_minority.png         ← Figure 5
    ├── figure6_communication.png    ← Figure 6
    ├── figure7_convergence.png      ← Figure 7
    └── byzantine_robustness.png
```

---

## Running Tests

```bash
python tests/test_dbec.py
python tests/test_saa.py
python tests/test_privacy.py
```

All tests should pass with output ending in `All X tests passed ✓`.

---

## Hardware Notes

Physical measurements (Table V) were taken on:
- **Device**: Raspberry Pi 5 (BCM2712 Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X)
- **Power meter**: Fnirsi FNB58, 100 Hz sampling
- **Precision**: float32
- **Batch size**: 1 (single-flow online inference)
- **Framework**: PyTorch 2.1

Software simulation of the remaining 40 clients used an NVIDIA RTX A5000.
