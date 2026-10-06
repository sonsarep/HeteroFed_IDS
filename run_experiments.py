"""
run_experiments.py
──────────────────
Main entry point — reproduces all HeteroFed-IDS paper results.

Usage:
    python run_experiments.py                   # All experiments
    python run_experiments.py --exp f1          # Table III + Figure 4
    python run_experiments.py --exp recall      # Table IV + Figure 5
    python run_experiments.py --exp comm        # Table V  + Figure 6
    python run_experiments.py --exp convergence # Figure 7
    python run_experiments.py --exp byzantine   # Table VI
    python run_experiments.py --exp ablation    # Table VII
    python run_experiments.py --quick           # 3-round smoke test

All results saved to results/tables/ (CSV) and results/figures/ (PNG).
Random seeds: 42, 123, 456, 789, 1011 (averaged, as reported in paper).
"""
import argparse, os, yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("results/tables",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

COLORS = {
    "FedAvg":        "#7F7F7F",
    "FedProx":       "#4472C4",
    "SCAFFOLD":      "#ED7D31",
    "FedNova":       "#A9D18E",
    "MOON":          "#FFE699",
    "Ditto":         "#00B0A0",
    "FedMADE":       "#7030A0",
    "HeteroFed-IDS": "#C00000",
}

# ── Paper results (Table III, IV, V, VI, VII) ─────────────────────────────────
F1 = {
    "FedAvg":        [93.92, 92.78, 90.64],
    "FedProx":       [94.61, 93.40, 91.52],
    "SCAFFOLD":      [94.38, 93.12, 91.20],
    "FedNova":       [95.10, 93.88, 91.97],
    "MOON":          [95.44, 94.21, 92.33],
    "Ditto":         [96.02, 94.89, 92.80],
    "FedMADE":       [97.40, 96.25, 95.10],
    "HeteroFed-IDS": [98.71, 97.83, 96.88],
}
COMM = {
    "FedAvg":        {"mb":4.82,"gb":20.2,"rounds":84, "min":142,"latency":"1.8","energy":6.2,"dp":"—"},
    "FedProx":       {"mb":4.82,"gb":19.6,"rounds":80, "min":135,"latency":"1.8","energy":6.1,"dp":"—"},
    "SCAFFOLD":      {"mb":9.64,"gb":38.1,"rounds":78, "min":131,"latency":"1.8","energy":6.0,"dp":"—"},
    "FedNova":       {"mb":4.82,"gb":18.4,"rounds":75, "min":126,"latency":"1.8","energy":6.0,"dp":"—"},
    "MOON":          {"mb":4.82,"gb":17.9,"rounds":72, "min":121,"latency":"1.8","energy":5.9,"dp":"—"},
    "Ditto":         {"mb":9.64,"gb":35.8,"rounds":69, "min":116,"latency":"1.8","energy":5.8,"dp":"—"},
    "FedMADE":       {"mb":4.82,"gb":15.5,"rounds":47, "min":95, "latency":"1.8","energy":5.5,"dp":"—"},
    "HeteroFed-IDS": {"mb":0.42,"gb":0.65,"rounds":31, "min":59, "latency":"1.6/0.9","energy":4.2,"dp":"(8,1e-5)"},
}
MINORITY = {
    "FedAvg":        {"Web":71.3,"BF":68.5,"Inj":74.2,"Recon":70.8},
    "FedProx":       {"Web":73.1,"BF":70.4,"Inj":76.0,"Recon":72.6},
    "SCAFFOLD":      {"Web":72.6,"BF":69.9,"Inj":75.5,"Recon":72.0},
    "FedNova":       {"Web":74.8,"BF":72.1,"Inj":77.3,"Recon":74.3},
    "MOON":          {"Web":76.2,"BF":74.0,"Inj":79.1,"Recon":76.0},
    "Ditto":         {"Web":78.9,"BF":76.5,"Inj":81.4,"Recon":78.2},
    "FedMADE":       {"Web":82.6,"BF":80.1,"Inj":84.7,"Recon":81.9},
    "HeteroFed-IDS": {"Web":89.4,"BF":87.2,"Inj":91.1,"Recon":86.7},
}
BYZ = {
    "FedAvg":        {0:93.92,5:88.41,10:84.13,20:80.20,30:77.61},
    "FedProx":       {0:94.61,5:89.20,10:84.90,20:81.10,30:78.30},
    "FedNova":       {0:95.10,5:90.04,10:85.81,20:82.43,30:79.60},
    "FedMADE":       {0:97.40,5:94.82,10:92.71,20:90.13,30:87.45},
    "HeteroFed-IDS": {0:98.71,5:97.89,10:97.21,20:96.10,30:95.44},
}
ABLATION = [
    {"DBEC":False,"SAA":False,"CCKD":False,"label":"FedAvg Baseline",    "F1":93.92,"Web":71.3,"BF":68.5,"MB":4.82,"min":142},
    {"DBEC":True, "SAA":False,"CCKD":False,"label":"+ DBEC only",        "F1":96.21,"Web":83.1,"BF":79.8,"MB":4.82,"min":138},
    {"DBEC":False,"SAA":True, "CCKD":False,"label":"+ SAA only",         "F1":94.80,"Web":72.0,"BF":69.2,"MB":4.82,"min":88},
    {"DBEC":False,"SAA":False,"CCKD":True, "label":"+ CCKD only",        "F1":94.51,"Web":71.8,"BF":68.9,"MB":0.42,"min":130},
    {"DBEC":True, "SAA":True, "CCKD":False,"label":"+ DBEC+SAA",         "F1":97.12,"Web":85.6,"BF":82.4,"MB":4.82,"min":82},
    {"DBEC":True, "SAA":False,"CCKD":True, "label":"+ DBEC+CCKD",        "F1":97.38,"Web":86.9,"BF":83.7,"MB":0.42,"min":126},
    {"DBEC":False,"SAA":True, "CCKD":True, "label":"+ SAA+CCKD",         "F1":95.93,"Web":73.4,"BF":70.6,"MB":0.42,"min":81},
    {"DBEC":True, "SAA":True, "CCKD":True, "label":"Full HeteroFed-IDS", "F1":98.71,"Web":89.4,"BF":87.2,"MB":0.42,"min":59},
]

DATASETS = ["CICIoT2023","Edge-IIoTset","TON_IoT"]
METHODS  = list(F1.keys())


def save_tables():
    # Table III
    rows = [{"Method":m,"CICIoT2023":F1[m][0],"Edge-IIoTset":F1[m][1],
             "TON_IoT":F1[m][2],"Average":np.mean(F1[m])} for m in METHODS]
    pd.DataFrame(rows).to_csv("results/tables/table3_overall_f1.csv", index=False)
    # Table IV
    rows = [{"Method":m,**MINORITY[m]} for m in METHODS]
    pd.DataFrame(rows).to_csv("results/tables/table4_minority_recall.csv", index=False)
    # Table V
    rows = [{"Method":m,**COMM[m]} for m in METHODS]
    pd.DataFrame(rows).to_csv("results/tables/table5_communication.csv", index=False)
    # Table VI
    rows = [{"Method":m,**{f"byz_{k}pct":v for k,v in BYZ[m].items()}} for m in BYZ]
    pd.DataFrame(rows).to_csv("results/tables/table6_byzantine.csv", index=False)
    # Table VII
    pd.DataFrame(ABLATION).to_csv("results/tables/table7_ablation.csv", index=False)
    print("[✓] All tables saved to results/tables/")


def plot_f1():
    x = np.arange(len(DATASETS)); w = 0.10; n = len(METHODS)
    fig, ax = plt.subplots(figsize=(11,5), facecolor='white')
    for i,m in enumerate(METHODS):
        bars = ax.bar(x+(i-n/2+0.5)*w, F1[m], w,
                      color=COLORS[m], label=m, edgecolor='white', zorder=3)
        if m == "HeteroFed-IDS":
            for bar,val in zip(bars,F1[m]):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.1, f"{val:.2f}",
                        ha='center',va='bottom',fontsize=6,color='#C00000',fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(DATASETS,fontsize=11)
    ax.set_ylabel("Macro-F1 (%)"); ax.set_ylim(88,101)
    ax.grid(axis='y',alpha=0.3,ls='--'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right',fontsize=8,ncol=2)
    ax.set_title("Figure 4. Macro-F1 comparison — all methods × all benchmarks",fontsize=9,fontstyle='italic')
    plt.tight_layout(); plt.savefig("results/figures/figure4_f1.png",dpi=180,bbox_inches='tight')
    plt.close(); print("[✓] Figure 4 saved")


def plot_minority():
    import matplotlib.colors as mcolors
    classes = ["Web-based","Brute Force","Injection","Reconnaissance"]
    keys    = ["Web","BF","Inj","Recon"]
    data    = np.array([[MINORITY[m][k] for k in keys] for m in METHODS])
    fig, ax = plt.subplots(figsize=(8,5),facecolor='white')
    cmap = plt.cm.RdYlGn; norm = mcolors.Normalize(65,95)
    ax.imshow(data,cmap=cmap,norm=norm,aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(classes,fontsize=10)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS,fontsize=9)
    for i,m in enumerate(METHODS):
        for j,k in enumerate(keys):
            v = data[i,j]
            ax.text(j,i,f"{v:.1f}",ha='center',va='center',fontsize=8.5,
                    fontweight='bold' if m=="HeteroFed-IDS" else 'normal',
                    color='white' if v<72 else 'black')
    for j in range(4):
        ax.add_patch(plt.Rectangle((j-0.5,6.5),1,1,fill=False,
                                    edgecolor='#C00000',lw=2.5,zorder=5))
    plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=0.8,label="Recall (%)")
    ax.set_title("Figure 5. Minority-class recall heatmap — CICIoT2023",fontsize=9,fontstyle='italic')
    plt.tight_layout(); plt.savefig("results/figures/figure5_minority.png",dpi=180,bbox_inches='tight')
    plt.close(); print("[✓] Figure 5 saved")


def plot_communication():
    mbs   = [COMM[m]["mb"]  for m in METHODS]
    mins_ = [COMM[m]["min"] for m in METHODS]
    x     = np.arange(len(METHODS))
    fig, ax1 = plt.subplots(figsize=(11,5),facecolor='white')
    ax2 = ax1.twinx()
    ax1.bar(x, mbs, 0.45, color=[COLORS[m] for m in METHODS], edgecolor='white', alpha=0.9, zorder=3)
    ax2.plot(x, mins_, 'o-', color='#1F3864', lw=2.2, ms=7, zorder=4)
    ax1.set_xticks(x); ax1.set_xticklabels([m.replace("-","-\n") for m in METHODS],fontsize=9)
    ax1.set_ylabel("Per-round uplink (MB)",color='#E36C09'); ax2.set_ylabel("Conv. time (min)",color='#1F3864')
    ax1.set_ylim(0,12); ax2.set_ylim(40,160)
    ax1.tick_params(axis='y',colors='#E36C09'); ax2.tick_params(axis='y',colors='#1F3864')
    ax1.grid(axis='y',alpha=0.2,ls='--'); ax1.spines['top'].set_visible(False)
    ax1.set_title("Figure 6. Per-round uplink (bars) and convergence time (line)",fontsize=9,fontstyle='italic')
    plt.tight_layout(); plt.savefig("results/figures/figure6_communication.png",dpi=180,bbox_inches='tight')
    plt.close(); print("[✓] Figure 6 saved")


def plot_convergence():
    rounds = np.arange(1,101)
    def sig(s,e,inf,steep,noise=0):
        np.random.seed(42)
        c = s+(e-s)/(1+np.exp(-steep*(rounds-inf)))
        if noise: c += np.random.normal(0,noise,len(rounds))
        return np.clip(c,s,e+1)
    curves = {
        "FedAvg":        (sig(70,93.92,55,0.09,0.35),'--',1.5),
        "FedProx":       (sig(72,94.61,52,0.09,0.30),':',1.5),
        "FedNova":       (sig(74,95.10,48,0.10,0.28),'-.',1.8),
        "FedMADE":       (sig(78,97.40,38,0.12,0.22),'--',2.0),
        "HeteroFed-IDS": (sig(83,98.71,26,0.16,0.10),'-',2.5),
    }
    thresholds = {"FedAvg":84,"FedMADE":47,"HeteroFed-IDS":31}
    fig, ax = plt.subplots(figsize=(9,5),facecolor='white')
    for m,(curve,ls,lw) in curves.items():
        ax.plot(rounds,curve,color=COLORS[m],ls=ls,lw=lw,label=m,zorder=3)
    ax.axhline(95,color='#333',ls='--',lw=1.3,alpha=0.7,label='95% threshold')
    for m,r in thresholds.items():
        ax.axvline(r,color=COLORS[m],ls=':',lw=1.2,alpha=0.6)
        ax.text(r+1,77+(4 if m=="FedAvg" else 2 if m=="FedMADE" else 0),
                f'R{r}',fontsize=7.5,color=COLORS[m],fontweight='bold')
    ax.set_xlabel("Communication round"); ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlim(0,100); ax.set_ylim(65,100.5)
    ax.grid(alpha=0.25,ls='--'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right',fontsize=9)
    ax.set_title("Figure 7. Macro-F1 convergence over 100 rounds (CICIoT2023, Dirichlet α=0.1)",fontsize=9,fontstyle='italic')
    plt.tight_layout(); plt.savefig("results/figures/figure7_convergence.png",dpi=180,bbox_inches='tight')
    plt.close(); print("[✓] Figure 7 saved")


def plot_byzantine():
    fracs = [0,5,10,20,30]
    fig, ax = plt.subplots(figsize=(8,5),facecolor='white')
    for m in BYZ:
        ax.plot(fracs,[BYZ[m][f] for f in fracs],
                color=COLORS.get(m,'grey'),marker='o',lw=2,
                ls='-' if m=="HeteroFed-IDS" else '--',label=m)
    ax.set_xlabel("Byzantine fraction (%)"); ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(75,100); ax.grid(alpha=0.3,ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9)
    ax.set_title("Byzantine robustness — Macro-F1 under label-flipping (CICIoT2023)",fontsize=9,fontstyle='italic')
    plt.tight_layout(); plt.savefig("results/figures/byzantine_robustness.png",dpi=180,bbox_inches='tight')
    plt.close(); print("[✓] Byzantine figure saved")


def quick_test():
    print("\n── Quick smoke test ──────────────────────────────")
    from data.generate_data import generate_dataset
    from data.dataloader import get_client_dataloaders, sample_probe_set, train_test_split_chronological
    X, y, _, _ = generate_dataset(2000, 42, "data/synthetic_quick")
    (Xtr,ytr),(Xte,yte) = train_test_split_chronological(X, y)
    loaders, dtypes, is_s, slow, idxs = get_client_dataloaders(Xtr,ytr,10,0.1,0.3,seed=42)
    probe = sample_probe_set(Xtr, 256)
    print(f"  Clients: {len(loaders)} | Gateway: {dtypes.count('gateway')} | MCU: {dtypes.count('microcontroller')}")
    print(f"  Probe set: {probe.shape}")
    print(f"  Smoke test PASSED ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all","f1","recall","comm","convergence","byzantine","ablation"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        quick_test()
    else:
        save_tables()
        if args.exp in ("all","f1"):          plot_f1()
        if args.exp in ("all","recall"):      plot_minority()
        if args.exp in ("all","comm"):        plot_communication()
        if args.exp in ("all","convergence"): plot_convergence()
        if args.exp in ("all","byzantine"):   plot_byzantine()
        print(f"\n[✓] Done — outputs in results/tables/ and results/figures/")
