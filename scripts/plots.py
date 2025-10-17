# dd_vs_ddultra_plots.py
# Draws (score distribution, t-SNE) panels per target for Deep Docking vs DD-Ultra.
# Colors: Deep Docking = blue, DD-Ultra = green, 1st-percentile threshold = red dashed.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, entropy   # H = -sum p log p  (Shannon)
# If you need to compute t-SNE embeddings from features, import:
# from sklearn.manifold import TSNE
from rdkit import Chem
import pandas as pd

# pip install rdkit-pypi scikit-learn scipy
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import entropy as H_shannon

def smiles_to_morgan_bits(smiles_list, n_bits=2048, radius=2):
    """ECFP4-like (Morgan r=2) binary fingerprints for each SMILES."""
    fps = []
    keep_idx = []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
        fps.append(bv)
        keep_idx.append(i)
    # Convert RDKit ExplicitBitVect → numpy uint8 array (0/1)
    arr = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, bv in enumerate(fps):
        onbits = list(bv.GetOnBits())
        arr[i, onbits] = 1
    return arr, fps, np.array(keep_idx, dtype=int)

def tanimoto_distance_matrix(fps):
    """NxN matrix with (1 - Tanimoto) distances from RDKit bitvectors."""
    n = len(fps)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i, i] = 0.0
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        D[i, i+1:] = 1.0 - np.asarray(sims, dtype=np.float32)
        D[i+1:, i] = D[i, i+1:]
    return D

def tsne_from_fps(fps, method="tanimoto", random_state=42, perplexity=30):
    """
    Build a 2-D t-SNE embedding from fingerprints.
    method:
      - "tanimoto": compute pairwise (1 - Tanimoto) and run TSNE(metric='precomputed')
      - "jaccard":  run TSNE directly on 0/1 arrays with metric='jaccard'
    """
    if method == "tanimoto":
        D = tanimoto_distance_matrix(fps)
        tsne = TSNE(n_components=2, metric='precomputed', perplexity=perplexity,
                    init='random', learning_rate='auto', random_state=random_state)
        XY = tsne.fit_transform(D)
    else:  # 'jaccard' (Jaccard == Tanimoto for binary sets)
        # Convert to 0/1 numpy arrays if not already
        # (if you have the bit arrays from smiles_to_morgan_bits, pass those instead)
        raise NotImplementedError("Pass the binary arrays to use metric='jaccard'.")
    return XY

def shannon_entropy_2d(points, bins=50, eps=1e-12):
    """H = -∑ p log p over a 2-D histogram of the embedding (nats)."""
    H2d, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=bins)
    p = H2d.ravel().astype(float)
    p /= (p.sum() + eps)
    p = p[p > 0]
    return float(H_shannon(p))

def _kde(x, grid=None, bw_method="scott", factor=1.5):
    x = np.asarray(x, dtype=float)
    if grid is None:
        xmin, xmax = np.min(x), np.max(x)
        pad = 0.05 * (xmax - xmin + 1e-12)
        grid = np.linspace(xmin - pad, xmax + pad, 512)

    # scale Scott/Silverman by `factor`
    if isinstance(bw_method, str):
        if bw_method == "scott":
            bw = lambda kde: kde.scotts_factor() * factor
        elif bw_method == "silverman":
            bw = lambda kde: kde.silverman_factor() * factor
        else:
            raise ValueError("bw_method must be 'scott', 'silverman', scalar, or callable")
    elif np.isscalar(bw_method):
        # direct scalar factor; multiply by extra factor if desired
        bw = bw_method * factor
    else:
        # callable provided; wrap to multiply
        bw = (lambda f: (lambda kde: f(kde) * factor))(bw_method)

    kde = gaussian_kde(x, bw_method=bw)
    return grid, kde(grid)

def _kde_old(x, grid=None, bw_method="scott"):
    """Return (grid, density) KDE for 1D array x."""
    x = np.asarray(x, dtype=float)
    if grid is None:
        xmin, xmax = np.percentile(x, [0.5, 99.5])
        pad = 0.05 * (xmax - xmin + 1e-12)
        grid = np.linspace(xmin - pad, xmax + pad, 512)
    kde = gaussian_kde(x, bw_method=bw_method)
    return grid, kde(grid)

def plot_dd_vs_ddultra(
    targets,
    train_scores=None,
    train_percentile=1.0,
    kde_bw="scott",
    s_scatter=8,
    figsize=(10, 14),
    suptitle="Deep Docking vs DD-Ultra across targets",
):
    """
    Parameters
    ----------
    targets : list of dicts, each with keys:
        {
          "name": str,
          "dd_scores": 1D array-like (top hits; docking scores),
          "ultra_scores": 1D array-like,
          "dd_tsne": array-like shape (N, 2),  # 2D embedding for DD hits
          "ultra_tsne": array-like shape (M, 2) # 2D embedding for DD-Ultra hits
        }
    train_scores : 1D array-like of randomly selected training compounds' docking scores
                   (for computing the 1st percentile threshold). If None, no threshold line.
    train_percentile : float
        Percentile (e.g., 1.0) to mark with a red dashed line on score plots.
    kde_bw : str or float
        Bandwidth for gaussian_kde ("scott" or "silverman" or numeric).
    bins_2d : int
        Number of bins per axis when computing 2D-entropy in t-SNE space.
    s_scatter : int
        Marker size for t-SNE scatter points.
    figsize : tuple
        Figure size.
    suptitle : str
        Figure super title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : dict with keys "score_axes" and "tsne_axes", each a list of Axes.
    """
    n = len(targets)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n, 1, figure=fig, wspace=0.25, hspace=0.35)

    score_axes, tsne_axes = [], []

    # Precompute the red threshold if training scores provided
    red_thresh = None
    if train_scores is not None and len(train_scores) > 0:
        train_scores = np.asarray(train_scores, dtype=float)
        red_thresh = np.percentile(train_scores, train_percentile)
    red_thresh = -8.5

    for i, T in enumerate(targets):
        name = T["name"]
        dd_scores = np.asarray(T["dd_scores"], dtype=float)
        ultra_scores = np.asarray(T["ultra_scores"], dtype=float)
        H_dd = T["H_dd"]
        H_ultra = T["H_ultra"]

        dd_mask = dd_scores != 0
        ultra_mask = ultra_scores != 0

        dd_scores = dd_scores[dd_mask]
        ultra_scores = ultra_scores[ultra_mask]

        # ── Left: docking score distributions (KDEs + optional threshold)
        axL = fig.add_subplot(gs[i, 0])
        score_axes.append(axL)

        # KDE curves
        grid_dd, dens_dd = _kde(dd_scores, bw_method=kde_bw)
        grid_ul, dens_ul = _kde(ultra_scores, bw_method=kde_bw)

        axL.plot(grid_dd, dens_dd, label="Random Docking", color="C0", lw=2)
        axL.plot(grid_ul, dens_ul, label="DD-Ultra", color="C2", lw=2)

        # Add red dashed threshold line
        if red_thresh is not None:
            axL.axvline(red_thresh, color="red", ls="--", lw=1.8, label="Threshold")

        axL.set_title(f"{name} — Docking Score Distribution", fontsize=12)
        axL.set_xlabel("Docking Score")
        axL.set_ylabel("Density")
        axL.grid(alpha=0.2)
        axL.legend(fontsize=10, loc="best")

        # ── Right: t-SNE scatter with entropy (diversity) annotation
        # axR = fig.add_subplot(gs[i, 1])
        # tsne_axes.append(axR)
        # dd_tsne = np.asarray(T["dd_tsne"], dtype=float)
        # ultra_tsne = np.asarray(T["ultra_tsne"], dtype=float)

        # axR.scatter(dd_tsne[:, 0], dd_tsne[:, 1], s=s_scatter, alpha=0.7, label="Deep Docking", color="C0")
        # axR.scatter(ultra_tsne[:, 0], ultra_tsne[:, 1], s=s_scatter, alpha=0.7, label="DD-Ultra", color="C2")

        # axR.set_title(f"{name} — t-SNE (H_dd={H_dd:.2f}, H_ultra={H_ultra:.2f})", fontsize=10)
        # axR.set_xlabel("t-SNE 1")
        # axR.set_ylabel("t-SNE 2")
        # axR.grid(alpha=0.2)
        # axR.legend(frameon=False, fontsize=9, loc="best")

    #fig.suptitle(suptitle, fontsize=16, y=0.995)
    return fig, {"score_axes": score_axes, "tsne_axes": tsne_axes}


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
            # cannot measure beyond the available ranking
            ks_used.append(K_eff)
            ef_vals.append(np.nan)
            tp_at_k.append(np.nan)
            continue
        
        tp = int(cumsum_actives[K_eff - 1])
        if K == 10:
            print(scores)
        #print(tp/K_eff)
        ef = (tp / K_eff) / max(base_frac, 1e-12)
        ks_used.append(K_eff)
        ef_vals.append(ef)
        tp_at_k.append(tp)

    return ks_used, ef_vals, tp_at_k, base_frac, N_baseline

import numpy as np
import matplotlib.pyplot as plt

def plot_enrichment_factor(
    ultra_scores_ranked,
    threshold=-8.5,
    Ks=(100_000, 1_000_000, 10_000_000, None),  # None => "all"
    baseline_scores=None,  # random 1M from library, if you have it
    total_size=None,
    title="Enrichment factor (EF) at selected K (DD-Ultra)"
):
    ultra_sorted = np.asarray(ultra_scores_ranked, float)
    #ultra_sorted = np.sort(ultra_scores_ranked)  # more negative = better

    # ---- compute EF for the requested K values
    ul_K, ul_EF, ul_TP, base_frac, N_base = compute_ef_at_k(
        ultra_sorted, threshold, Ks, baseline_scores=baseline_scores, total_size=total_size
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


# -------------------------------
# Example usage (with mock data):
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Pretend we have 5 targets; replace these with your real arrays
    targets = []
    name = "PGK2"
    # Mock top-2000 docking scores (negative "better")
    dd_exp = pd.read_pickle(f'results/enamine_10B_pgk2_cmp21_h2o_vina/random_1M_test.pkl')
    dd_scores = dd_exp['test']['dock_scores']
    dd_smiles = dd_exp['test']['smiles']
    strict_thresh = np.nanpercentile(dd_scores, 0.01)  # 0.01% == 0.01 percentile
    print("0.01% cutoff (more negative is better):", strict_thresh)

    ultra_exp = pd.read_csv(f'results/enamine_10B_pgk2_cmp21_h2o_vina/final_dock_proba_res.csv')
    ultra_exp = ultra_exp.sort_values("proba", ascending=False)
    ultra_exp.to_csv(f'results/enamine_10B_pgk2_cmp21_h2o_vina/final_dock_proba_res_sorted.csv', index=False)
    ultra_scores = ultra_exp['dock_score'].values
    ultra_smiles = ultra_exp['smiles'].values

    # --- build embeddings and entropies for your two sets ---
    # 1) Fingerprints as both numpy bit arrays (for Jaccard) and RDKit vectors (for Tanimoto)
    #dd_bits, dd_fps, _ = smiles_to_morgan_bits(dd_smiles, n_bits=2048, radius=2)
    #ultra_bits, ultra_fps, _ = smiles_to_morgan_bits(ultra_smiles, n_bits=2048, radius=2)

    # Option A (recommended for chemical fingerprints): Tanimoto distances
    #dd_tsne  = tsne_from_fps(dd_fps, method="tanimoto", perplexity=30, random_state=42)
    #ultra_tsne = tsne_from_fps(ultra_fps, method="tanimoto", perplexity=30, random_state=42)

    dd_tsne = None
    ultra_tsne = None
    # Option B (equivalent on binary sets): directly with Jaccard on 0/1 matrices
    # from sklearn.manifold import TSNE
    # dd_tsne = TSNE(n_components=2, metric='jaccard', init='random',
    #                learning_rate='auto', perplexity=30, random_state=42).fit_transform(dd_bits)
    # ultra_tsne = TSNE(n_components=2, metric='jaccard', init='random',
    #                   learning_rate='auto', perplexity=30, random_state=42).fit_transform(ultra_bits)

    # 2) Diversity: Shannon entropy of the 2-D occupancy (tune bins for stability vs. resolution)
    # H_dd     = shannon_entropy_2d(dd_tsne,    bins=50)
    # H_ultra  = shannon_entropy_2d(ultra_tsne, bins=50)
    H_dd = None
    H_ultra = None

    #print(f"H_dd (nats)   : {H_dd:.3f}")
    #print(f"H_ultra (nats): {H_ultra:.3f}")

    targets.append({
        "name": name,
        "dd_scores": dd_scores,
        "ultra_scores": ultra_scores,
        "dd_tsne": dd_tsne,
        "ultra_tsne": ultra_tsne,
        "H_dd": H_dd,
        "H_ultra": H_ultra,
    })

    # Randomly selected training compounds' scores (for the red threshold line)
    #train_scores = rng.normal(loc=-7.5, scale=0.8, size=100_000)

    # fig, _ = plot_dd_vs_ddultra(
    #     targets,
    #     train_scores=None,
    #     train_percentile=1.0,  # 1st percentile
    #     kde_bw="scott",
    #     s_scatter=6,
    #     figsize=(6, 4),
    #     suptitle="Comparative performance of Random Docking and DD-Ultra",
    # )
    # fig.savefig(f'plots/dd_vs_ddultra_pgk2_cmp21_h2o.png',dpi=300, bbox_inches="tight")
    # plt.show()

    Ks = (10, 100, 1000, 10_000, 100_000, 1_000_000)  # None = 'all'
    threshold = strict_thresh

    # If you *don't* have the full library, pass a random training pool to estimate baseline:
    # baseline_scores = np.asarray(train_scores)  # random library sample
    baseline_scores = dd_scores  # or provide your random sample here

    fig, ax, res = plot_enrichment_factor(
        ultra_scores,
        threshold=threshold, Ks=Ks,
        baseline_scores=baseline_scores,  # recommended if full library not available
        total_size=None,
        title="EF of virtual actives at Top-K for PGK2"
    )
    fig.savefig("plots/ef_vs_topk.png", dpi=300, bbox_inches="tight")
