import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF


def main():
    # 1. Setup output paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "results", "figures")
    tbl_dir = os.path.join(base_dir, "results", "tables")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # 2. Load EMG dataset
    emg_file = "EMG_musclesynergies.csv"
    emg_path = os.path.join(data_dir, emg_file)
    if not os.path.exists(emg_path):
        emg_path = emg_file

    df_emg = pd.read_csv(emg_path)

    time_col = df_emg.columns[0]
    emg_cols = df_emg.columns[1:]

    time_vec = df_emg[time_col].values
    V_raw = df_emg[emg_cols].values  # shape: (969 timepoints, 12 muscles)

    clean_muscle_names = [
        col.split(":")[0].replace("->MAV", "").strip() for col in emg_cols
    ]

    # ---------------------------------------------------------
    # Problem 4(a): Subplots of 12 Muscles & Bilateral TA Overlay
    # ---------------------------------------------------------
    mask_20s = time_vec <= 20.0
    time_20s = time_vec[mask_20s]
    V_20s = V_raw[mask_20s, :]

    # Plot 1: 12-Channel Subplot Grid
    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True, dpi=300)
    fig.suptitle(
        "Problem 4(a): EMG Time-Series Envelope (First 20 Seconds)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    axes_flat = axes.flatten()
    for i in range(12):
        ax = axes_flat[i]
        ax.plot(time_20s, V_20s[:, i], color="darkblue", linewidth=1.0)
        ax.set_title(clean_muscle_names[i], fontsize=10, fontweight="bold")
        ax.set_ylabel("MAV (Volts)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes_flat[-2].set_xlabel("Time (sec)", fontsize=10)
    axes_flat[-1].set_xlabel("Time (sec)", fontsize=10)

    plt.tight_layout()
    fig4a_path = os.path.join(fig_dir, "problem4a_emg_signals.png")
    plt.savefig(fig4a_path)
    plt.close()

    # Plot 2: Bilateral TA Comparison (Overlay on Single Plot)
    r_ta_idx = [
        i for i, name in enumerate(clean_muscle_names) if "R TIBIALIS" in name
    ][0]
    l_ta_idx = [
        i for i, name in enumerate(clean_muscle_names) if "L TIBIALIS" in name
    ][0]

    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(
        time_20s,
        V_20s[:, r_ta_idx],
        label="Right Tibialis Anterior (R TA)",
        color="tab:blue",
        linewidth=1.5,
        alpha=0.85,
    )
    plt.plot(
        time_20s,
        V_20s[:, l_ta_idx],
        label="Left Tibialis Anterior (L TA)",
        color="tab:orange",
        linewidth=1.5,
        alpha=0.85,
    )

    plt.title(
        "Problem 4(a): Bilateral TA Overlay Showing Phase-Shifted Alternating Activations",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Time (sec)", fontsize=11)
    plt.ylabel("MAV (Volts)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    fig4a_overlay_path = os.path.join(
        fig_dir, "problem4a_bilateral_ta_overlay.png"
    )
    plt.savefig(fig4a_overlay_path)
    plt.close()

    # ---------------------------------------------------------
    # Problem 4(b): Non-Negative Factorization (NMF, 4 Components)
    # ---------------------------------------------------------
    V_matrix = V_raw.T  # (12 muscles x 969 timepoints)

    n_synergies = 4
    nmf_model = NMF(
        n_components=n_synergies, init="random", random_state=42, max_iter=5000
    )
    W_spatial = nmf_model.fit_transform(V_matrix)
    H_temporal = nmf_model.components_

    V_hat = W_spatial @ H_temporal

    # Metrics
    vaf = (1.0 - np.sum((V_matrix - V_hat) ** 2) / np.sum(V_matrix**2)) * 100.0
    v_mean = np.mean(V_matrix)
    r2 = (
        1.0 - np.sum((V_matrix - V_hat) ** 2) / np.sum((V_matrix - v_mean) ** 2)
    ) * 100.0

    metrics_path = os.path.join(tbl_dir, "problem4b_nmf_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("NMF Model Fit Metrics (4 Muscle Synergies)\n")
        f.write("=" * 45 + "\n")
        f.write(f"Variance Accounted For (VAF): {vaf:.2f}%\n")
        f.write(f"Coefficient of Determination (R^2): {r2:.2f}%\n")

    # ---------------------------------------------------------
    # Problem 4(c): Heatmap of Spatial Synergy Basis Matrix W
    # ---------------------------------------------------------
    W_norm = W_spatial / np.max(W_spatial, axis=0, keepdims=True)

    df_W = pd.DataFrame(
        W_norm,
        index=clean_muscle_names,
        columns=[f"Synergy {i+1}" for i in range(n_synergies)],
    )

    plt.figure(figsize=(8, 7), dpi=300)
    sns.heatmap(
        df_W,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Relative Synergy Weight"},
        linewidths=0.5,
    )
    plt.title(
        "Problem 4(c): Spatial Muscle Synergy Weights (Basis Matrix W)",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("Muscle Channel", fontsize=11)
    plt.xlabel("Latent Synergy Component", fontsize=11)
    plt.tight_layout()

    fig4c_path = os.path.join(fig_dir, "problem4c_synergies_heatmap.png")
    plt.savefig(fig4c_path)
    plt.close()

    # ---------------------------------------------------------
    # PRINT RESULTS DIRECTLY TO COLAB/TERMINAL CONSOLE
    # ---------------------------------------------------------
    print("=" * 65)
    print("                PROBLEM 4 EXECUTION RESULTS                 ")
    print("=" * 65)

    print("\n--- Problem 4(a) Results ---")
    print(f"12-Channel Subplot figure saved to: {fig4a_path}")
    print(f"Bilateral TA Overlay figure saved to: {fig4a_overlay_path}")

    print("\n--- Problem 4(b) Results ---")
    print(f"(i)  Variance Accounted For (VAF): {vaf:.2f}%")
    print(f"(ii) R^2 Metric:                   {r2:.2f}%")

    print("\n--- Problem 4(c) Spatial Basis Matrix W (Normalized) ---")
    print(df_W.round(3).to_string())
    print(f"\nHeatmap plot saved to: {fig4c_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()