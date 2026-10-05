"""Module 2: fixed-snapshot clustering, PCA, and centered SVD.

Run: python code/04_unsupervised.py
Never downloads data or replaces Module 1 outputs. IDs and discovery methods are
kept separately for interpretation; only eight numeric columns enter any fit.
"""
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures/module2"
REPORT = ROOT / "reports/module2"
DATA = ROOT / "data/processed/module2"
FEATURES = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax",
            "pl_orbeccen", "st_teff", "st_mass", "st_rad"]
LOG_FEATURES = [c for c in FEATURES if c != "pl_orbeccen"]
LABELS = ["Radius", "Mass", "Period", "Semi-major axis", "Eccentricity",
          "Star temperature", "Star mass", "Star radius"]
COLORS = ["#4477aa", "#ee7733", "#228833", "#cc6677", "#aa3377",
          "#66ccee", "#8877aa", "#999933"]
SEED = 42


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def preview(frame, name, title):
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    table = ax.table(cellText=frame.head(6).round(3).astype(str).values,
                     colLabels=["z log radius", "z log mass", "z log period", "z log axis",
                                "z eccentricity", "z log star T", "z log star M", "z log star R"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.75)
    ax.set_title(title, pad=22)
    save(fig, name)


def relabel(labels, frame):
    """Stable display IDs ordered by the clusters' median planet radius."""
    order = sorted(np.unique(labels), key=lambda c: frame.loc[labels == c, "pl_rade"].median())
    return np.array([{old: i + 1 for i, old in enumerate(order)}[v] for v in labels])


def concepts():
    """Original explanatory images, explicitly separate from observations."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.add_patch(plt.Circle((0.28, 0.55), 0.19, color="#eeb64b"))
    ax.add_patch(plt.Circle((0.34, 0.55), 0.045, color="#243c5d"))
    ax.text(0.28, 0.27, "Planet crosses the stellar disk", ha="center")
    t = np.linspace(0, 1, 180)
    flux = 0.67 - 0.13 * np.exp(-((t - 0.5) / 0.12)**8)
    ax.plot(0.55 + t * 0.4, flux, color="#4477aa", lw=3)
    ax.text(0.75, 0.3, "Small, repeated dips in observed light", ha="center")
    ax.set(xlim=(0, 1), ylim=(0, 1), title="Transit detection | conceptual illustration, not to scale")
    ax.axis("off"); save(fig, "intro_transit")
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(t, np.sin(t), color="#4477aa", lw=3)
    ax.axhline(0, color="#9ba6b4", lw=1)
    ax.annotate("Star moving away: redshift", xy=(1.57, 1), xytext=(2.7, 1.3),
                arrowprops={"arrowstyle": "->"})
    ax.annotate("Star moving toward us: blueshift", xy=(4.71, -1), xytext=(0.1, -1.5),
                arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Orbital phase (schematic)", ylabel="Stellar line-of-sight velocity (schematic)",
           title="Radial-velocity detection | conceptual illustration", ylim=(-1.8, 1.8))
    save(fig, "intro_velocity")
    toy = np.array([[-2, -1], [-1.8, -.5], [-1.5, -1.2], [1.5, 1], [2, .5], [2.3, 1.3]])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(*toy.T, c=[COLORS[0]]*3+[COLORS[1]]*3, s=90)
    centers = np.vstack([toy[:3].mean(0), toy[3:].mean(0)])
    ax.scatter(*centers.T, marker="X", s=180, color="#222222", label="Centroids")
    ax.legend(); ax.set(xlabel="Illustrative feature 1", ylabel="Illustrative feature 2",
                       title="Partitional clustering | synthetic teaching example")
    save(fig, "concept_partition")
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(linkage(toy, method="average"), labels=list("ABCDEF"), ax=ax)
    ax.set(ylabel="Average-linkage Euclidean distance (toy example)",
           title="Hierarchical clustering | synthetic teaching example")
    save(fig, "concept_hierarchy")
    cloud = np.random.default_rng(SEED).multivariate_normal([0, 0], [[2, 1.6], [1.6, 2]], size=80)
    for projection in [False, True]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(*cloud.T, s=20, color="#4477aa", alpha=.6)
        v = np.array([1., 1.]) / np.sqrt(2)
        ax.plot([-3, 3], [-3, 3], color="#ee7733", label="PC1: maximum variance")
        if projection:
            proj = np.outer(cloud @ v, v)
            for point, dest in zip(cloud[::8], proj[::8]):
                ax.plot([point[0], dest[0]], [point[1], dest[1]], "--", color="#999999")
            ax.scatter(*proj[::8].T, color="#ee7733", s=35)
        else:
            ax.plot([-1.5, 1.5], [1.5, -1.5], color="#228833", label="PC2: orthogonal direction")
        ax.set(xlabel="Centered illustrative feature 1", ylabel="Centered illustrative feature 2",
               title=("Projection onto PC1" if projection else "Principal directions") + " | synthetic teaching example")
        ax.legend(fontsize=9); save(fig, "concept_projection" if projection else "concept_pca")


def main():
    for folder in [FIG, REPORT, DATA]: folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 14})
    source = ROOT / "data/processed/exoplanets_clean.csv"
    original = pd.read_csv(source)
    numeric = original[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    valid &= (numeric[LOG_FEATURES] > 0).all(axis=1) & numeric.pl_orbeccen.between(0, 1)
    rows = original.loc[valid].sort_values("pl_name", kind="stable").reset_index().rename(columns={"index": "source_row"})
    raw = rows[FEATURES].copy()
    transformed = raw.copy()
    transformed[LOG_FEATURES] = np.log10(transformed[LOG_FEATURES])
    scaler = StandardScaler()
    x = scaler.fit_transform(transformed)
    assert np.isfinite(x).all() and np.all(np.linalg.norm(x, axis=1) > 1e-12)
    assert np.allclose(x.mean(0), 0, atol=1e-12) and np.allclose(x.std(0), 1)
    columns = ["z_log10_"+c if c in LOG_FEATURES else "z_"+c for c in FEATURES]
    matrix = pd.DataFrame(x, columns=columns)
    raw.to_csv(DATA / "physical_features.csv", index=False)
    matrix.to_csv(DATA / "model_matrix.csv", index=False)
    matrix.head(12).to_csv(DATA / "model_sample.csv", index=False)
    rows[["source_row", "pl_name", "discoverymethod"]].to_csv(DATA / "row_metadata.csv", index=False)
    preview(matrix, "model_preview", "Model input: standardized numeric features only (first six rows)")
    bias = pd.concat([original.discoverymethod.value_counts(normalize=True).rename("snapshot_share"),
                      rows.discoverymethod.value_counts(normalize=True).rename("retained_share")], axis=1).fillna(0)
    bias.to_csv(REPORT / "selection_by_method.csv")
    prep = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_rows": len(original), "retained_rows": len(rows), "excluded_rows": len(original)-len(rows),
            "features": FEATURES, "log10_features": LOG_FEATURES, "imputation": "none",
            "sample_order": "planet name ascending; zero-based source_row",
            "scaler_mean": dict(zip(FEATURES, scaler.mean_)), "scaler_scale": dict(zip(FEATURES, scaler.scale_)),
            "missing_per_feature": numeric.isna().sum().to_dict(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}}
    (REPORT / "preparation.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
    print(f"Prepared {x.shape}; excluded {prep['excluded_rows']} incomplete or invalid rows", flush=True)

    pca = PCA(svd_solver="full").fit(x)
    scores = pca.transform(x)
    # Fix sign ambiguity consistently across the independently computed PCA/SVD.
    for i in range(x.shape[1]):
        if pca.components_[i, np.argmax(abs(pca.components_[i]))] < 0:
            pca.components_[i] *= -1; scores[:, i] *= -1
    u, singular, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    for i in range(x.shape[1]):
        if vt[i] @ pca.components_[i] < 0: vt[i] *= -1; u[:, i] *= -1
    assert np.allclose(pca.explained_variance_, singular**2/(len(x)-1))
    assert np.allclose(scores, u * singular) and np.allclose(vt, pca.components_)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    q90 = int(np.searchsorted(cumulative, .9)+1)
    pcs = [f"PC{i+1}" for i in range(x.shape[1])]
    pd.DataFrame(pca.components_.T, index=FEATURES, columns=pcs).to_csv(REPORT / "pca_loadings.csv")
    pd.DataFrame(scores, columns=pcs).to_csv(DATA / "pca_scores.csv", index=False)
    pd.DataFrame(scores, columns=pcs).assign(discoverymethod=rows.discoverymethod).groupby(
        "discoverymethod").median().to_csv(REPORT / "pca_scores_by_method.csv")
    correlations = np.corrcoef(np.column_stack([x, scores]).T)[:8, 8:]
    pd.DataFrame(correlations, index=FEATURES, columns=pcs).to_csv(REPORT / "pca_variable_correlations.csv")
    spectrum = pd.DataFrame({"component": np.arange(1, 9), "eigenvalue": pca.explained_variance_,
                             "explained_share": pca.explained_variance_ratio_, "cumulative_share": cumulative,
                             "singular_value": singular})
    errors = []
    for q in range(1, 9):
        reconstruction = (u[:, :q] * singular[:q]) @ vt[:q]
        error = np.linalg.norm(x-reconstruction)/np.linalg.norm(x)
        assert np.isclose(error**2, 1-cumulative[q-1], atol=1e-12)
        errors.append(float(error))
    spectrum["relative_frobenius_error"] = errors
    spectrum.to_csv(REPORT / "pca_svd_spectrum.csv", index=False)

    # Full-data fits; fixed common 2,000-row subset for all silhouette comparisons.
    evaluation = np.sort(np.random.default_rng(SEED).choice(len(x), min(2000, len(x)), replace=False))
    pd.DataFrame({"model_row": evaluation}).to_csv(REPORT / "silhouette_rows.csv", index=False)
    euclidean = pairwise_distances(x[evaluation], metric="euclidean")
    cosine = pairwise_distances(x[evaluation], metric="cosine")
    np.fill_diagonal(euclidean, 0); np.fill_diagonal(cosine, 0)
    tree = linkage(pdist(x, metric="cosine"), method="average")
    pd.DataFrame(tree, columns=["left", "right", "cosine_merge_height", "size"]).to_csv(REPORT / "linkage.csv", index=False)
    km_labels = {}; hc_labels = {}; comparisons = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=30, random_state=SEED).fit(x)
        a = relabel(km.labels_, raw)
        b = relabel(cut_tree(tree, n_clusters=k).ravel(), raw)
        km_labels[k] = a; hc_labels[k] = b
        comparisons.append({"k": k, "inertia": float(km.inertia_),
            "kmeans_silhouette_euclidean": float(silhouette_samples(euclidean, a[evaluation], metric="precomputed").mean()),
            "hierarchical_silhouette_cosine": float(silhouette_samples(cosine, b[evaluation], metric="precomputed").mean()),
            "same_k_adjusted_rand": float(adjusted_rand_score(a, b)),
            "kmeans_min_cluster_size": int(np.bincount(a)[1:].min()),
            "hierarchical_min_cluster_size": int(np.bincount(b)[1:].min())})
        print(comparisons[-1], flush=True)
    metrics = pd.DataFrame(comparisons)
    metrics.to_csv(REPORT / "clustering_metrics.csv", index=False)
    best_k = int(metrics.loc[metrics.kmeans_silhouette_euclidean.idxmax(), "k"])
    best_h = int(metrics.loc[metrics.hierarchical_silhouette_cosine.idxmax(), "k"])
    gaps = [{"k": k, "lower": float(tree[-k, 2]), "upper": float(tree[-k+1, 2]),
             "gap": float(tree[-k+1, 2]-tree[-k, 2])} for k in range(2, 9)]
    gap_choice = max(gaps, key=lambda d: d["gap"])
    a = km_labels[best_k]; b = hc_labels[best_h]
    assignments = rows[["source_row", "pl_name", "discoverymethod"]].copy()
    assignments["kmeans_cluster"] = a; assignments["hierarchical_cluster"] = b
    assignments.to_csv(REPORT / "cluster_assignments.csv", index=False)
    profiles = raw.assign(cluster=a).groupby("cluster").median()
    profiles.insert(0, "n", pd.Series(a).value_counts().sort_index())
    profiles.to_csv(REPORT / "kmeans_profiles.csv")
    hprofiles = raw.assign(cluster=b).groupby("cluster").median()
    hprofiles.insert(0, "n", pd.Series(b).value_counts().sort_index())
    hprofiles.to_csv(REPORT / "hierarchical_profiles.csv")
    contingency = pd.crosstab(pd.Series(a, name="kmeans"), pd.Series(b, name="hierarchical"))
    contingency.to_csv(REPORT / "cluster_agreement.csv")
    boot_ari = []
    for seed in range(5):
        rng = np.random.default_rng(100+seed)
        subset = rng.choice(len(x), int(.8*len(x)), replace=False)
        fitted = KMeans(n_clusters=best_k, n_init=20, random_state=seed).fit(x[subset])
        boot_ari.append(float(adjusted_rand_score(a, fitted.predict(x))))
    # Redundant orbital features can overweight one physical direction.
    keep = [i for i,c in enumerate(FEATURES) if c != "pl_orbsmax"]
    sensitivity = KMeans(n_clusters=best_k, n_init=30, random_state=SEED).fit_predict(x[:, keep])
    summary = {"best_kmeans_k": best_k, "best_hierarchical_k": best_h,
        "silhouette_evaluation_n": len(evaluation), "evaluation_seed": SEED,
        "k_range": list(range(2, 9)), "kmeans_n_init": 30,
        "hierarchical_metric": "cosine", "hierarchical_linkage": "average",
        "largest_late_merge_gap": gap_choice, "late_merge_gaps": gaps,
        "chosen_partitions_ari": float(adjusted_rand_score(a,b)),
        "subsample_80pct_ari": boot_ari,
        "omit_semimajor_axis_ari": float(adjusted_rand_score(a,sensitivity)),
        "pc1_pc2_share": float(cumulative[1]), "components_for_90pct": q90,
        "retained_share": float(cumulative[q90-1]), "rank_q90_relative_error": errors[q90-1],
        "pca_svd_max_score_difference": float(np.max(abs(scores-u*singular))),
        "full_rank_relative_error": errors[-1]}
    (REPORT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(metrics.k, metrics.inertia, "o-", color=COLORS[0]); axes[0].set(xlabel="Number of clusters k", ylabel="WCSS (inertia)", title="K-means elbow | all retained rows")
    axes[1].plot(metrics.k, metrics.kmeans_silhouette_euclidean, "o-", color=COLORS[0], label="K-means / Euclidean")
    axes[1].plot(metrics.k, metrics.hierarchical_silhouette_cosine, "s--", color=COLORS[1], label="Average linkage / cosine")
    axes[1].set(xlabel="Number of clusters k", ylabel="Mean silhouette", title=f"Common {len(evaluation):,}-row evaluation subset")
    axes[1].legend(fontsize=9); fig.tight_layout(); save(fig, "cluster_selection")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax,k in zip(axes,[2,3,4]):
        for label in np.unique(km_labels[k]):
            mask=km_labels[k]==label
            ax.scatter(scores[mask,0], scores[mask,1], s=6, alpha=.4, color=COLORS[label-1], rasterized=True)
        ax.set(xlabel="PC1 score", ylabel="PC2 score", title=f"K-means k = {k}")
    fig.suptitle("Full eight-feature fits, displayed in a common PCA projection", y=1.04)
    fig.tight_layout(); save(fig,"kmeans_comparison")
    fig, ax = plt.subplots(figsize=(11, 5))
    threshold = (tree[-best_h, 2]+tree[-best_h+1, 2])/2
    dendrogram(tree, truncate_mode="lastp", p=30, show_leaf_counts=True, color_threshold=threshold, ax=ax)
    ax.axhline(threshold, color="#aa3377", ls="--", label=f"k={best_h} cut at {threshold:.3f}")
    ax.set(xlabel="Contracted branches (parentheses show number of planets)", ylabel="Average cosine dissimilarity",
           title=f"Agglomerative dendrogram | all {len(x):,} retained planets; last 30 branches")
    ax.legend(); save(fig,"dendrogram")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, labs, title in [(axes[0],a,f"K-means k={best_k}"),(axes[1],b,f"Cosine hierarchy k={best_h}")]:
        for label in np.unique(labs):
            mask=labs==label
            ax.scatter(scores[mask,0],scores[mask,1],s=8,alpha=.45,color=COLORS[label-1],label=f"C{label}: {mask.sum():,}")
        ax.set(xlabel="PC1 score",ylabel="PC2 score",title=title); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig,"cluster_projection")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(np.log10(raw.pl_bmasse),np.log10(raw.pl_rade),c=[COLORS[i-1] for i in a],s=10,alpha=.4)
    ax.set(xlabel="log10 planet mass (Earth masses)",ylabel="log10 planet radius (Earth radii)",title=f"Physical context | eight-feature k-means, k={best_k}")
    save(fig,"cluster_mass_radius")
    sil = silhouette_samples(euclidean, a[evaluation], metric="precomputed")
    fig, ax=plt.subplots(figsize=(9,5)); offset=0
    for label in np.unique(a):
        vals=np.sort(sil[a[evaluation]==label]); positions=np.arange(offset,offset+len(vals))
        ax.fill_betweenx(positions,0,vals,color=COLORS[label-1],alpha=.8,label=f"C{label}")
        offset += len(vals)+30
    ax.axvline(sil.mean(),ls="--",color="#222222",label=f"Mean {sil.mean():.3f}")
    ax.set(xlabel="Euclidean silhouette",ylabel="Evaluation observations grouped by cluster",title=f"Selected k-means partition | k={best_k}")
    ax.legend(); save(fig,"silhouette_detail")
    fig, axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(range(1,9),100*pca.explained_variance_ratio_,color=COLORS[0]); axes[0].set(xlabel="Principal component",ylabel="Explained variance (%)",title="PCA scree plot")
    axes[1].plot(range(1,9),100*cumulative,"o-",color=COLORS[0]); axes[1].axhline(90,ls="--",color="#aa3377")
    axes[1].axvline(q90,ls=":",color="#aa3377"); axes[1].set(xlabel="Components retained",ylabel="Cumulative variance (%)",title=f"90% rule retains {q90} components",ylim=(0,105))
    fig.tight_layout(); save(fig,"pca_variance")
    fig,ax=plt.subplots(figsize=(9,5.5))
    categories=rows.discoverymethod.where(rows.discoverymethod.isin(["Transit","Radial Velocity","Microlensing"]),"Other")
    for i,cat in enumerate(["Transit","Radial Velocity","Microlensing","Other"]):
        mask=categories==cat
        ax.scatter(scores[mask,0],scores[mask,1],s=10,alpha=.4,color=COLORS[i],label=f"{cat}: {mask.sum():,}")
    ax.set(xlabel=f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)",ylabel=f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)",title="PCA scores | discovery method held out of the fit")
    ax.legend(fontsize=9); save(fig,"pca_projection")
    fig,ax=plt.subplots(figsize=(10,5))
    im=ax.imshow(correlations[:,:4],vmin=-1,vmax=1,cmap="RdBu_r",aspect="auto")
    ax.set(yticks=range(8),yticklabels=LABELS,xticks=range(4),xticklabels=pcs[:4],title="Original transformed features vs principal-component scores")
    for i in range(8):
        for j in range(4): ax.text(j,i,f"{correlations[i,j]:.2f}",ha="center",va="center",color="white" if abs(correlations[i,j])>.65 else "black")
    fig.colorbar(im,ax=ax,label="Pearson correlation (not eigenvector coefficient)"); fig.tight_layout(); save(fig,"pca_correlations")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(range(1,9),singular,"o-",color=COLORS[0]); axes[0].set(xlabel="Singular-vector index",ylabel="Singular value",title="Centered SVD spectrum")
    axes[1].plot(range(1,9),errors,"o-",color=COLORS[1]); axes[1].axvline(q90,ls=":",color="#aa3377")
    axes[1].set(xlabel="Reconstruction rank q",ylabel="Relative Frobenius error",title=f"Rank-{q90} error = {errors[q90-1]:.3f}")
    fig.tight_layout(); save(fig,"svd_reconstruction")
    concepts()
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
