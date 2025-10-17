
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_ef_at_k(sorted_scores, threshold, Ks, baseline_scores=None, total_size=None):
    scores = np.asarray(sorted_scores)
    # label actives
    is_active = (scores < threshold).astype(np.int32)

    # baseline active fraction
    if baseline_scores is not None:
        b = np.asarray(baseline_scores)
        base_frac = float(np.mean(b < threshold))
        N_baseline = total_size if (total_size is not None) else len(b)
    else:
        base_frac = float(np.mean(is_active))
        N_baseline = len(scores)

    # cumulative actives to enable fast TP@K
    cumsum_actives = np.cumsum(is_active)
    ks_used, ef_vals, tp_at_k = [], [], []

    for K in Ks:
        if K is None:  # 'all'
            K_eff = len(scores)
        else:
            K_eff = int(K)

        if K_eff > len(scores):
            ks_used.append(K_eff)
            ef_vals.append(np.nan)
            tp_at_k.append(np.nan)
            continue

        tp = int(cumsum_actives[K_eff - 1])
        ef = (tp / K_eff) / max(base_frac, 1e-12)
        ks_used.append(K_eff)
        ef_vals.append(ef)
        tp_at_k.append(tp)

    return ks_used, ef_vals, tp_at_k, base_frac, N_baseline


def plot_enrichment_factor(
    ultra_scores_ranked,
    threshold=-8.5,
    Ks=(100_000, 1_000_000, 10_000_000, None),  # None => "all"
    baseline_scores=None,  
    total_size=None,
    title="Enrichment factor (EF) at selected K (DD-Ultra)"
):
    ultra_scores = np.asarray(ultra_scores_ranked, float)
    #ultra_sorted = np.sort(ultra_scores_ranked)  # more negative = better

    # ---- compute EF for the requested K values
    ul_K, ul_EF, ul_TP, base_frac, N_base = compute_ef_at_k(
        ultra_scores, threshold, Ks, baseline_scores=baseline_scores, total_size=total_size
    )

    # keep only Ks we could actually compute (drop NaNs / overly large K)
    Ks_eff, EFs_eff, TPs_eff = [], [], []
    labels = []
    for k, ef, tp in zip(ul_K, ul_EF, ul_TP):
        if np.isnan(ef):
            continue
        Ks_eff.append(k)
        EFs_eff.append(ef)
        TPs_eff.append(tp)
        labels.append(f"{k:,}")

    # ---- bar plot
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)

    cmap = plt.get_cmap('tab10')          # or mpl.colormaps['tab10'] on newer MPL
    tol_muted = [
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#332288",  # indigo
    ]

    # use as many colors as needed
    colors = [tol_muted[i % len(tol_muted)] for i in range(len(EFs_eff))]

    x = np.arange(len(EFs_eff))
    bars = ax.bar(x, EFs_eff, color=colors, edgecolor="black", linewidth=0.6)

    # value labels on top of bars
    # for xi, bi, ef in zip(x, bars, EFs_eff):
    #     ax.annotate(f"{ef:.2f}",
    #                 xy=(xi, bi.get_height()),
    #                 xytext=(0, 4), textcoords="offset points",
    #                 ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Enrichment Factor")
    ax.set_title(title, pad=6)
    ax.grid(axis="y", alpha=0.25)

    # baseline caption
    # ax.text(0.01, 0.02,
    #         f"Baseline active fraction = {base_frac:.4f} (N≈{N_base})\nThreshold: score < {threshold}",
    #         transform=ax.transAxes, fontsize=8)

    return fig, ax, {
        "ultra": {"K": Ks_eff, "EF": EFs_eff, "TP": TPs_eff, "labels": labels},
        "baseline_active_fraction": base_frac,
        "baseline_N": N_base
    }


if __name__ == "__main__":
    dd_exp = pd.read_pickle(f'results/enamine_10B_pgk2_cmp21_h2o_vina/random_1M_test.pkl')
    rand_scores = dd_exp['test']['dock_scores']

    ultra_exp = pd.read_csv(f'results/enamine_10B_pgk2_cmp21_h2o_vina/final_dock_proba_res.csv')
    ultra_exp = ultra_exp.sort_values("proba", ascending=False)
    ultra_exp.to_csv(f'results/enamine_10B_pgk2_cmp21_h2o_vina/final_dock_proba_res_sorted.csv', index=False)
    ultra_scores = ultra_exp['dock_score'].values

    strict_thresh = np.nanpercentile(rand_scores, 0.01)  # 0.01% == 0.01 percentile

    Ks = (10, 100, 1000, 10_000, 100_000, 1_000_000)  # None = 'all'
    threshold = -8.5

    baseline_scores = rand_scores 

    fig, ax, res = plot_enrichment_factor(
        ultra_scores,
        threshold=strict_thresh, Ks=Ks,
        baseline_scores=baseline_scores,  
        total_size=None,
        title="EF of virtual actives at Top-K for PGK2"
    )
    fig.savefig("plots/ef_vs_topk.png", dpi=300, bbox_inches="tight")
