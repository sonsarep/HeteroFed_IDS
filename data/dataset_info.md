# Dataset Information

This repository does **not** include the raw datasets due to their size.  
Download them from the official sources below and place them in `data/real/`.

---

## CICIoT2023 (Primary Benchmark)

| Property | Value |
|---|---|
| **URL** | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| **Size** | ~47 million flows |
| **Devices** | 105 IoT devices |
| **Features** | 47 flow-level (extracted with CICFlowMeter) |
| **Attack classes** | 33 |
| **Reference** | E. C. P. Neto et al., *Sensors*, vol. 23, no. 13, p. 5941, Jun. 2023. DOI: 10.3390/s23135941 |

**Preparation:**
```bash
# After downloading, place CSV files in:
data/real/CICIoT2023/

# Then run:
python data/preprocess.py --dataset CICIoT2023 --in data/real/CICIoT2023/ --out data/processed/
```

---

## Edge-IIoTset

| Property | Value |
|---|---|
| **URL** | https://ieee-dataport.org/documents/edge-iiotset |
| **DOI** | 10.21227/mbc1-1h68 |
| **Size** | ~1.9 million samples |
| **Features** | 61 |
| **Attack classes** | 14 |
| **Reference** | M. A. Ferrag et al., *IEEE Access*, vol. 10, pp. 40281–40306, Mar. 2022. DOI: 10.1109/ACCESS.2022.3165809 |

**Preparation:**
```bash
python data/preprocess.py --dataset Edge-IIoTset --in data/real/Edge-IIoTset/ --out data/processed/
```

---

## TON_IoT

| Property | Value |
|---|---|
| **URL** | https://research.unsw.edu.au/projects/toniot-datasets |
| **Size** | ~22 million records |
| **Features** | Network + OS + telemetry |
| **Attack classes** | 9 |
| **Reference** | A. Alsaedi et al., *IEEE Access*, vol. 8, pp. 165130–165150, Sep. 2020. DOI: 10.1109/ACCESS.2020.3022862 |

**Preparation:**
```bash
python data/preprocess.py --dataset TON_IoT --in data/real/TON_IoT/ --out data/processed/
```

---

## Synthetic Dataset (No Download Required)

For quick testing without downloading the full datasets, use the built-in generator:

```bash
python data/generate_data.py --samples 200000 --out data/synthetic/
```

This generates a synthetic CICIoT2023-style dataset with:
- 47 flow-level features (same names and ranges as CICIoT2023)
- 33 attack classes with realistic class imbalance
- Per-device traffic signatures for DBEC validation
- Correct minority/majority class split (minority = <5% of samples)

---

## Data Split

For all experiments, the **chronological 70/30 split** is used:
- Training: first 70% of flows by timestamp
- Testing: last 30% of flows by timestamp

This prevents future flows from leaking into training, as required for fair IDS evaluation.
The Dirichlet α=0.1 label skew is applied **after** the chronological split,
independently per device type.
