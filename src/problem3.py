import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA


def main():
    # 1. Setup directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "results", "figures")

    os.makedirs(fig_dir, exist_ok=True)

    # 2. Load EEG dataset (.pkl file)
    eeg_path = os.path.join(data_dir, "eegdata.pkl")
    df_eeg = pd.read_pickle(eeg_path)

    # Extract time vector (first column) and electrode matrix (subsequent columns)
    time_vec = df_eeg.iloc[:, 0].values
    eeg_signals = df_eeg.iloc[:, 1:].values  # Shape: (timepoints, 97 electrodes)

    # ---------------------------------------------------------
    # Problem 3(a): Fit FastICA (10 components) & Identify Artifact
    # ---------------------------------------------------------
    n_components = 10
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)

    # FastICA transforms X (timepoints, electrodes) into S (timepoints, components)
    # X = S * A^T where A is mixing matrix
    S = ica.fit_transform(eeg_signals)  # Independent components
    A = ica.mixing_  # Mixing matrix: (97 electrodes, 10 components)

    # Find component with prominent periodic spikes (eyeblink artifact)
    # Eyeblink artifacts exhibit high variance and periodic extreme peak amplitudes
    comp_peak_counts = []
    comp_max_amps = []

    for i in range(n_components):
        comp = S[:, i]
        # Standardize for peak detection scale comparison
        comp_norm = (comp - np.mean(comp)) / np.std(comp)
        comp_max_amps.append(np.max(np.abs(comp_norm)))

    # Component with maximum absolute amplitude peak is eyeblink artifact
    eyeblink_comp_idx = int(np.argmax(comp_max_amps))

    # Plot 10 Independent Components (10 Subplots)
    fig, axes = plt.subplots(
        n_components, 1, figsize=(12, 14), sharex=True, dpi=300
    )
    fig.suptitle(
        "Problem 3(a): 10 Extracted Independent Components (ICA)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for i in range(n_components):
        ax = axes[i]
        color = "crimson" if i == eyeblink_comp_idx else "navy"
        ax.plot(time_vec, S[:, i], color=color, linewidth=0.8)
        label_text = f"IC {i+1}"
        if i == eyeblink_comp_idx:
            label_text += " (Eyeblink Artifact)"
        ax.set_ylabel(
            label_text,
            fontsize=9,
            fontweight="bold" if i == eyeblink_comp_idx else "normal",
        )
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (sec)", fontsize=11)
    plt.tight_layout()

    fig3a_path = os.path.join(fig_dir, "problem3a_ica_components.png")
    plt.savefig(fig3a_path)
    plt.close()

    # ---------------------------------------------------------
    # Problem 3(b): Remove Eyeblink Artifact & Plot Electrode 1
    # ---------------------------------------------------------
    # Zero out the eyeblink component column in independent sources S
    S_cleaned = S.copy()
    S_cleaned[:, eyeblink_comp_idx] = 0

    # Reconstruct electrode signal matrix: X_cleaned = S_cleaned * A^T + mean
    # Inverse transform via sklearn FastICA
    cleaned_signals = ica.inverse_transform(S_cleaned)

    # Electrode 1 data comparison (Original vs Cleaned)
    orig_electrode1 = eeg_signals[:, 0]
    cleaned_electrode1 = cleaned_signals[:, 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=300)

    # Original Electrode 1
    ax1.plot(time_vec, orig_electrode1, color="#1f77b4", linewidth=0.8)
    ax1.set_title(
        "Original EEG Data (Electrode 1)", fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("Voltage", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Cleaned Electrode 1
    ax2.plot(time_vec, cleaned_electrode1, color="#2ca02c", linewidth=0.8)
    ax2.set_title(
        f"Cleaned EEG Data without Eyeblink Artifact (IC {eyeblink_comp_idx+1} Removed)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Voltage", fontsize=10)
    ax2.set_xlabel("Time (sec)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    fig3b_path = os.path.join(fig_dir, "problem3b_cleaned_electrode1.png")
    plt.savefig(fig3b_path)
    plt.close()

    # ---------------------------------------------------------
    # PRINT RESULTS DIRECTLY TO COLAB/TERMINAL CONSOLE
    # ---------------------------------------------------------
    print("=" * 65)
    print("                PROBLEM 3 EXECUTION RESULTS                 ")
    print("=" * 65)

    print("\n--- Problem 3(a) Results ---")
    print(
        f"Loaded EEG matrix shape: {eeg_signals.shape} (Timepoints x Electrodes)"
    )
    print(f"Number of Independent Components Extracted: {n_components}")
    print(
        f"-> Eyeblink Artifact Component Identified: Component IC {eyeblink_comp_idx + 1}"
    )
    print(f"10 Component subplot figure saved to: {fig3a_path}")

    print("\n--- Problem 3(b) Results ---")
    print(
        f"Zeroed out IC {eyeblink_comp_idx + 1} across all 97 electrode channels."
    )
    print("Reconstructed signals using inverse ICA transform.")
    print(
        f"Electrode 1 comparison plot (Original vs Cleaned) saved to: {fig3b_path}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()