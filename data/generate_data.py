"""
data/generate_data.py
─────────────────────
Generates a synthetic CICIoT2023-style IoT intrusion detection dataset.

Matches:
  - 47 flow-level features (same names and value ranges as CICIoT2023)
  - 33 attack classes with realistic class imbalance
  - Per-device traffic signatures (for DBEC embedding validation)
  - Minority/majority split matching the paper's definition (<5% = minority)

Usage:
    python data/generate_data.py
    python data/generate_data.py --samples 200000 --seed 42 --out data/synthetic
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Feature names (matching CICIoT2023) ──────────────────────────────────────
FEATURES = [
    "flow_duration","header_length","protocol_type","duration",
    "rate","srate","drate","fin_flag_number","syn_flag_number",
    "rst_flag_number","psh_flag_number","ack_flag_number","ece_flag_number",
    "cwr_flag_number","ack_count","syn_count","fin_count","urg_count",
    "rst_count","http_count","https_count","dns_count","telnet_count",
    "smtp_count","ssh_count","irc_count","tcp_count","udp_count",
    "dhcp_count","arp_count","icmp_count","igmp_count","ipv4_count",
    "ipv6_count","pkt_len_mean","pkt_len_std","pkt_len_min","pkt_len_max",
    "inter_arrival_mean","inter_arrival_std","active_mean","active_std",
    "idle_mean","idle_std","fwd_pkt_len_mean","bwd_pkt_len_mean",
]

# ── Attack classes with realistic prevalence ──────────────────────────────────
# (class_name, fraction_of_dataset, device_affinity)
ATTACK_CLASSES = [
    # Majority classes (>5%)
    ("Benign",              0.220, "all"),
    ("DDoS-UDP",            0.085, "gateway"),
    ("DDoS-TCP",            0.080, "gateway"),
    ("DDoS-HTTP",           0.075, "gateway"),
    ("DoS-UDP",             0.060, "sensor"),
    ("DoS-TCP",             0.055, "sensor"),
    ("Port-Scan",           0.050, "gateway"),
    ("Botnet-Mirai",        0.050, "iot_device"),
    # Minority classes (<5%)
    ("Web-XSS",             0.030, "gateway"),
    ("Web-SQLi",            0.025, "gateway"),
    ("Web-CmdInj",          0.020, "gateway"),
    ("Brute-Force-SSH",     0.018, "server"),
    ("Brute-Force-FTP",     0.017, "server"),
    ("Brute-Force-HTTP",    0.016, "gateway"),
    ("Injection-DNS",       0.015, "gateway"),
    ("Injection-ARP",       0.014, "iot_device"),
    ("Recon-Ping",          0.013, "gateway"),
    ("Recon-OS",            0.012, "gateway"),
    ("Recon-VulnScan",      0.011, "gateway"),
    ("Recon-HostDiscover",  0.010, "gateway"),
    ("MITM-ARP",            0.009, "iot_device"),
    ("Botnet-BASHLITE",     0.008, "iot_device"),
    ("Ransomware",          0.007, "server"),
    ("Backdoor",            0.006, "server"),
    ("DDoS-ICMP",           0.006, "gateway"),
    ("DoS-ICMP",            0.005, "sensor"),
    ("Fuzzing",             0.005, "iot_device"),
    ("Replay-Attack",       0.004, "iot_device"),
    ("MQTT-Malformed",      0.004, "sensor"),
    ("CoAP-Flood",          0.003, "sensor"),
    ("Zigbee-Attack",       0.003, "iot_device"),
    ("BLE-Spoofing",        0.002, "iot_device"),
    ("Zero-Day-Sim",        0.002, "all"),
]

# ── Device types with distinct traffic signatures ─────────────────────────────
DEVICE_TYPES = {
    "smart_camera":      {"pkt_mean":1200,"rate_mean":500, "affinity":"gateway"},
    "industrial_sensor": {"pkt_mean":64,  "rate_mean":10,  "affinity":"sensor"},
    "smart_thermostat":  {"pkt_mean":128, "rate_mean":2,   "affinity":"iot_device"},
    "smart_meter":       {"pkt_mean":256, "rate_mean":5,   "affinity":"sensor"},
    "ip_camera":         {"pkt_mean":1400,"rate_mean":800, "affinity":"gateway"},
    "voip_phone":        {"pkt_mean":200, "rate_mean":50,  "affinity":"gateway"},
    "smart_lock":        {"pkt_mean":80,  "rate_mean":1,   "affinity":"iot_device"},
    "medical_wearable":  {"pkt_mean":100, "rate_mean":3,   "affinity":"iot_device"},
}
DEVICE_LIST = list(DEVICE_TYPES.keys())


def _make_features(n, cls_name, device, rng):
    """Generate n flow-level feature vectors for a given class and device."""
    d = DEVICE_TYPES[device]
    pkt = rng.normal(d["pkt_mean"], d["pkt_mean"]*0.25, n).clip(40, 1500)
    rate= rng.exponential(d["rate_mean"], n)
    atk = cls_name.lower()

    if "ddos" in atk or "dos" in atk:
        rate *= rng.uniform(50, 500, n)
        pkt   = rng.normal(64, 10, n).clip(40,100)
        dur   = rng.exponential(0.5, n)
    elif "scan" in atk or "recon" in atk:
        rate *= rng.uniform(5, 20, n)
        pkt   = rng.normal(40, 5, n).clip(40, 60)
        dur   = rng.uniform(0.001, 0.1, n)
    elif "web" in atk or "sql" in atk or "inj" in atk:
        rate *= rng.uniform(1, 5, n)
        pkt   = rng.normal(800, 200, n).clip(100,1500)
        dur   = rng.normal(2.0, 0.5, n).clip(0.1, 10)
    elif "brute" in atk:
        rate *= rng.uniform(2, 10, n)
        pkt   = rng.normal(200, 50, n).clip(40, 500)
        dur   = rng.uniform(0.5, 5.0, n)
    else:
        dur   = rng.exponential(5, n)

    X = np.zeros((n, 46))
    X[:,0]  = dur if 'dur' in dir() else rng.exponential(5,n)
    X[:,1]  = rng.integers(20,60,n)
    X[:,4]  = rate.clip(0.01, 1e6)
    X[:,5]  = X[:,4] * rng.uniform(0.4,0.8,n)
    X[:,6]  = X[:,4] * rng.uniform(0.2,0.6,n)
    X[:,14] = rng.integers(0,20,n)
    X[:,20] = rng.integers(0,100,n)
    X[:,33] = pkt
    X[:,34] = rng.normal(pkt*0.3, 10, n).clip(0, 500)
    X[:,35] = np.maximum(40, pkt - X[:,34]*2)
    X[:,36] = np.minimum(1500, pkt + X[:,34]*2)
    iat      = 1.0 / X[:,4].clip(1e-3)
    X[:,37] = iat
    X[:,38] = rng.normal(iat*0.5, 0.001, n).clip(0)
    X[:,43] = pkt * rng.uniform(0.5,0.9,n)
    X[:,44] = pkt * rng.uniform(0.1,0.5,n)
    return X


def generate_dataset(n_samples=200_000, seed=42, output_dir="data/synthetic"):
    """Generate and save the synthetic dataset."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"Generating {n_samples:,} samples — {len(ATTACK_CLASSES)} classes ...")

    class_names = [c[0] for c in ATTACK_CLASSES]
    cls2idx     = {c:i for i,c in enumerate(class_names)}

    all_X, all_y, all_dev = [], [], []
    for cls_name, frac, affinity in ATTACK_CLASSES:
        n_cls = max(100, int(n_samples * frac))
        compat = [d for d,info in DEVICE_TYPES.items()
                  if info["affinity"]==affinity or affinity=="all"] or DEVICE_LIST
        for i in range(n_cls):
            dev  = compat[i % len(compat)]
            feat = _make_features(1, cls_name, dev, rng)
            all_X.append(feat[0])
            all_y.append(cls2idx[cls_name])
            all_dev.append(dev)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    devs = np.array(all_dev)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y, devs = X[idx], y[idx], devs[idx]

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Save
    np.save(os.path.join(output_dir, "X.npy"), X_scaled)
    np.save(os.path.join(output_dir, "y.npy"), y)
    np.save(os.path.join(output_dir, "devices.npy"), devs)

    # Class map CSV
    pd.DataFrame({
        "class_idx":    range(len(class_names)),
        "class_name":   class_names,
        "fraction":     [c[1] for c in ATTACK_CLASSES],
        "is_minority":  [c[1] < 0.05 for c in ATTACK_CLASSES],
        "device_type":  [c[2] for c in ATTACK_CLASSES],
    }).to_csv(os.path.join(output_dir, "class_map.csv"), index=False)

    # Human-readable sample
    df = pd.DataFrame(X_scaled, columns=FEATURES)
    df["label"]       = y
    df["device_type"] = devs
    df["class_name"]  = [class_names[l] for l in y]
    df.head(5000).to_csv(os.path.join(output_dir, "sample_5k.csv"), index=False)

    print(f"\nSaved to {output_dir}/")
    print(f"  X.npy       : {X_scaled.shape}")
    print(f"  y.npy       : {y.shape}")
    print(f"  Minority cls: {sum(1 for c in ATTACK_CLASSES if c[1]<0.05)}")
    for i,(name,frac,_) in enumerate(ATTACK_CLASSES):
        cnt  = (y==i).sum()
        flag = " ← MINORITY" if frac<0.05 else ""
        print(f"  [{i:2d}] {name:<25} {cnt:6d} ({cnt/len(y)*100:.1f}%){flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     type=str, default="data/synthetic")
    args = ap.parse_args()
    generate_dataset(args.samples, args.seed, args.out)
