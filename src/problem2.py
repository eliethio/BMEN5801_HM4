import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main():
    # 1. Setup paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "results", "figures")
    tbl_dir = os.path.join(base_dir, "results", "tables")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # 2. Load proteome dataset
    proteome_file = (
        "breastcancerproteomes.csv"
        if os.path.exists(os.path.join(data_dir, "breastcancerproteomes.csv"))
        else "breastcancerproteome.csv"
    )
    df_proteome = pd.read_csv(os.path.join(data_dir, proteome_file))

    # Separate metadata columns from sample columns
    metadata_cols = [
        c
        for c in ["RefSeq_accession_number", "gene_symbol", "gene_name"]
        if c in df_proteome.columns
    ]
    sample_cols = [c for c in df_proteome.columns if c not in metadata_cols]

    # Transpose: rows = samples (83), cols = proteins (12,553)
    df_expr = df_proteome[sample_cols].T

    # Impute missing values (NaN) with column (protein) median
    df_imputed = df_expr.fillna(df_expr.median())

    # ---------------------------------------------------------
    # Problem 2(a): PCA (PC1 vs PC2)
    # ---------------------------------------------------------
    pca_10 = PCA(n_components=10)
    pcs_10 = pca_10.fit_transform(df_imputed)

    pca_df = pd.DataFrame(
        pcs_10[:, :2], columns=["PC1", "PC2"], index=df_imputed.index
    )
    var_exp = pca_10.explained_variance_ratio_ * 100
    cum_var_exp = np.cumsum(var_exp)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        color="skyblue",
        edgecolor="k",
        alpha=0.8,
        s=60,
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.2f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.2f}% Variance)")
    plt.title(
        "Problem 2(a): PCA of Breast Cancer Biopsy Proteomes (PC1 vs PC2)"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig2a_path = os.path.join(fig_dir, "problem2a_pca_scatterplot.png")
    plt.savefig(fig2a_path)
    plt.close()

    # ---------------------------------------------------------
    # Problem 2(b): Variance Summary Output
    # ---------------------------------------------------------
    metrics_path = os.path.join(tbl_dir, "problem2b_variance_summary.txt")
    with open(metrics_path, "w") as f:
        f.write("PC\tExplained Variance (%)\tCumulative Variance (%)\n")
        f.write("-" * 55 + "\n")
        for i in range(10):
            f.write(
                f"PC{i+1}\t{var_exp[i]:.4f}%\t\t\t{cum_var_exp[i]:.4f}%\n"
            )
        f.write("-" * 55 + "\n")
        f.write(
            f"Total Cumulative Explained Variance (10 PCs): {cum_var_exp[-1]:.2f}%\n"
        )

    # ---------------------------------------------------------
    # Problem 2(c): PCA Scatter Plot Colored by ER Status
    # ---------------------------------------------------------
    df_clinical = pd.read_csv(
        os.path.join(data_dir, "breastcancerclinical.csv")
    )

    # Map patient IDs cleanly
    df_clinical["patient_id"] = df_clinical["Sample"].str.replace(
        "TCGA-", "", regex=False
    )
    mapping = dict(zip(df_clinical["patient_id"], df_clinical["ER Status"]))

    er_labels = []
    for s in pca_df.index:
        if "CPTAC" in s:
            er_labels.append("Control")
        else:
            pid = s.split(".")[0]
            er_labels.append(mapping.get(pid, "Unknown"))

    pca_df["ER_Status"] = er_labels

    plt.figure(figsize=(8, 6), dpi=300)

    pos_mask = pca_df["ER_Status"] == "Positive"
    neg_mask = pca_df["ER_Status"] == "Negative"

    plt.scatter(
        pca_df.loc[pos_mask, "PC1"],
        pca_df.loc[pos_mask, "PC2"],
        color="red",
        label="Positive (ER+)",
        edgecolor="k",
        alpha=0.8,
        s=60,
    )
    plt.scatter(
        pca_df.loc[neg_mask, "PC1"],
        pca_df.loc[neg_mask, "PC2"],
        color="blue",
        label="Negative (ER-)",
        edgecolor="k",
        alpha=0.8,
        s=60,
    )

    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.2f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.2f}% Variance)")
    plt.title("Problem 2(c): PCA of Proteome Data by ER Status")
    plt.legend(title="ER Status", loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig2c_path = os.path.join(fig_dir, "problem2c_pca_er_status.png")
    plt.savefig(fig2c_path)
    plt.close()

    # ---------------------------------------------------------
    # PRINT RESULTS DIRECTLY TO COLAB/TERMINAL CONSOLE
    # ---------------------------------------------------------
    print("=" * 65)
    print("                PROBLEM 2 EXECUTION RESULTS                 ")
    print("=" * 65)

    print("\n--- Problem 2(a) Results ---")
    print(f"Sample Expression Matrix Shape: {df_imputed.shape} (Samples x Proteins)")
    print(f"PC1 Variance Explained: {var_exp[0]:.2f}%")
    print(f"PC2 Variance Explained: {var_exp[1]:.2f}%")
    print(f"Scatter plot saved to: {fig2a_path}")

    print("\n--- Problem 2(b) Results ---")
    print(f"{'PC':<8}{'Explained Variance (%)':<25}{'Cumulative Variance (%)':<25}")
    print("-" * 60)
    for i in range(10):
        print(f"PC{i+1:<6}{var_exp[i]:<25.4f}{cum_var_exp[i]:<25.4f}")
    print("-" * 60)
    print(f"Total Cumulative Explained Variance across 10 PCs: {cum_var_exp[-1]:.2f}%\n")

    print("--- Problem 2(c) Results ---")
    print("Mapped ER Status Breakdown for Proteome Samples:")
    print(pca_df["ER_Status"].value_counts().to_string())
    print(f"\nER Status Scatter plot saved to: {fig2c_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()