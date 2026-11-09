import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score


def main():
    # 1. Setup output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "results", "figures")
    tbl_dir = os.path.join(base_dir, "results", "tables")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # 2. Load PBMC3K single-cell dataset
    pbmc_path = os.path.join(data_dir, "PBMC3K.csv")
    if not os.path.exists(pbmc_path):
        pbmc_path = "PBMC3K.csv"

    df_pbmc = pd.read_csv(pbmc_path)

    cell_types = df_pbmc["CellType"].values
    X_features = df_pbmc.drop(columns=["CellType"]).values

    unique_cells = np.unique(cell_types)
    palette = sns.color_palette("tab10", len(unique_cells))
    color_dict = dict(zip(unique_cells, palette))

    # ---------------------------------------------------------
    # 1. PCA Baseline Projection (problem5_pca.png)
    # ---------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_features)
    pca_var = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(9, 7), dpi=300)
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=cell_types,
        palette=color_dict,
        s=35,
        alpha=0.75,
        edgecolor="none",
    )
    plt.title(
        "PCA Projection of PBMC3K Single-Cell Data",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(f"PC1 ({pca_var[0]:.2f}% Variance)")
    plt.ylabel(f"PC2 ({pca_var[1]:.2f}% Variance)")
    plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    fig_pca_path = os.path.join(fig_dir, "problem5_pca.png")
    plt.savefig(fig_pca_path)
    plt.close()

    # ---------------------------------------------------------
    # 2. t-SNE Parameter Sweep & Metrics Calculation
    # ---------------------------------------------------------
    params = [5, 15, 30, 60, 100]
    tsne_trust, tsne_sil = [], []
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.suptitle(
        "Problem 5: t-SNE Projections Across Perplexities [5, 15, 30, 60, 100]",
        fontsize=14,
        fontweight="bold",
    )
    axes_flat = axes.flatten()

    for idx, perp in enumerate(params):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X_features)

        tr_val = trustworthiness(X_features, X_tsne, n_neighbors=15)
        sil_val = silhouette_score(X_tsne, cell_types)
        tsne_trust.append(tr_val)
        tsne_sil.append(sil_val)

        ax = axes_flat[idx]
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            hue=cell_types,
            palette=color_dict,
            s=18,
            alpha=0.75,
            edgecolor="none",
            ax=ax,
            legend=(idx == 2),
        )
        ax.set_title(
            f"t-SNE (Perp={perp}) | Sil={sil_val:.3f} | Trust={tr_val:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes_flat[-1].axis("off")
    axes_flat[2].legend(
        title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()

    fig_tsne_path = os.path.join(fig_dir, "problem5_tsne_perplexities.png")
    plt.savefig(fig_tsne_path)
    plt.close()

    # ---------------------------------------------------------
    # 3. UMAP Parameter Sweep & Metrics Calculation
    # ---------------------------------------------------------
    umap_trust, umap_sil = [], []
    has_umap = False

    try:
        import umap

        has_umap = True

        fig_u, axes_u = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig_u.suptitle(
            "Problem 5: UMAP Projections Across n_neighbors [5, 15, 30, 60, 100]",
            fontsize=14,
            fontweight="bold",
        )
        axes_u_flat = axes_u.flatten()

        for idx, n_n in enumerate(params):
            reducer = umap.UMAP(n_neighbors=n_n, min_dist=0.3, random_state=42)
            X_umap = reducer.fit_transform(X_features)

            tr_val = trustworthiness(X_features, X_umap, n_neighbors=15)
            sil_val = silhouette_score(X_umap, cell_types)
            umap_trust.append(tr_val)
            umap_sil.append(sil_val)

            ax = axes_u_flat[idx]
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                hue=cell_types,
                palette=color_dict,
                s=18,
                alpha=0.75,
                edgecolor="none",
                ax=ax,
                legend=(idx == 2),
            )
            ax.set_title(
                f"UMAP (n_neighbors={n_n}) | Sil={sil_val:.3f} | Trust={tr_val:.3f}",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.grid(True, linestyle="--", alpha=0.4)

        axes_u_flat[-1].axis("off")
        axes_u_flat[2].legend(
            title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()
        fig_umap_path = os.path.join(
            fig_dir, "problem5_umap_neighbors_sweep.png"
        )
        plt.savefig(fig_umap_path)
        plt.close()
    except ImportError:
        pass

    # ---------------------------------------------------------
    # 4. Direct Side-by-Side Metrics Comparison Plots
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ax1.plot(
        params,
        tsne_trust,
        marker="o",
        label="t-SNE",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
    )
    if has_umap:
        ax1.plot(
            params,
            umap_trust,
            marker="s",
            label="UMAP",
            color="#ff7f0e",
            linewidth=2.5,
            markersize=7,
        )
    ax1.set_title(
        "Local Neighborhood Preservation (Trustworthiness)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Hyperparameter Value (Perplexity / n_neighbors)")
    ax1.set_ylabel("Trustworthiness Score")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(
        params,
        tsne_sil,
        marker="o",
        label="t-SNE",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
    )
    if has_umap:
        ax2.plot(
            params,
            umap_sil,
            marker="s",
            label="UMAP",
            color="#ff7f0e",
            linewidth=2.5,
            markersize=7,
        )
    ax2.set_title(
        "Cluster Separation Quality (Silhouette Score)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Hyperparameter Value (Perplexity / n_neighbors)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig_metrics_path = os.path.join(
        fig_dir, "problem5b_tsne_vs_umap_metrics.png"
    )
    plt.savefig(fig_metrics_path)
    plt.close()

    # ---------------------------------------------------------
    # 5. Export Summary Metrics Table & Print Outputs
    # ---------------------------------------------------------
    metrics_dict = {
        "Parameter": params,
        "t-SNE Trustworthiness": np.round(tsne_trust, 4),
        "t-SNE Silhouette": np.round(tsne_sil, 4),
    }
    if has_umap:
        metrics_dict["UMAP Trustworthiness"] = np.round(umap_trust, 4)
        metrics_dict["UMAP Silhouette"] = np.round(umap_sil, 4)

    df_metrics = pd.DataFrame(metrics_dict)
    tbl_path = os.path.join(tbl_dir, "problem5b_metrics_summary.csv")
    df_metrics.to_csv(tbl_path, index=False)

    print("=" * 72)
    print("                PROBLEM 5 EXECUTION RESULTS                 ")
    print("=" * 72)
    print(df_metrics.to_string(index=False))
    print(f"\nPCA plot saved to:      {fig_pca_path}")
    print(f"t-SNE plot saved to:    {fig_tsne_path}")
    print(f"Metrics plot saved to:  {fig_metrics_path}")
    print(f"Metrics table saved to: {tbl_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()