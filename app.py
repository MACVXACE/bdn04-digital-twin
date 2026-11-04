import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from src.signal_generator import generate_wetware_signal, inject_wetware_noise
from src.twin_engine import run_kalman_observer, calculate_rmse

st.set_page_config(page_title="BDN-04 Wetware Digital Twin", layout="wide")

plt.style.use("dark_background")

def render_plot(x, traces, ylabel="Amplitude (V)", height=2.6):
    fig, ax = plt.subplots(figsize=(10, height), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for label, y, color, alpha, lw, ls in traces:
        ax.plot(x, y, label=label, color=color, alpha=alpha, linewidth=lw, linestyle=ls)
    ax.set_ylabel(ylabel, fontsize=9, color="#fafafa")
    ax.grid(True, color="#262730", alpha=0.6, linewidth=0.6)
    ax.tick_params(colors="#a3a8b8", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#262730")
    if len(traces) > 1:
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.title("BDN-04 Biological Computing Digital Twin")
st.caption("Dual-State Observer & Real-Time Electrophysiology Reconstruction Engine")

st.sidebar.header("Biological Wetware Config")
spike_rate = st.sidebar.slider("Action Potential Firing Rate (Hz)", 1.0, 25.0, 8.0, 0.5)

st.sidebar.header("Electrode Noise & Artifacts")
noise_mode = st.sidebar.selectbox("Noise Profile", ["Pink (1/f) + Drift", "Pink (1/f) Only", "AWGN Only"])
snr_db = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", -5.0, 25.0, 7.0, 0.5)
drift_amp = st.sidebar.slider("Baseline Drift Amplitude", 0.0, 2.0, 0.6, 0.05) if "Drift" in noise_mode else 0.0
seed = st.sidebar.slider("RNG Seed", 1, 100, 42, 1)

st.sidebar.header("Twin Observer Tuning")
log_q_neural = st.sidebar.slider("Log10 Q (Neural Dynamics)", -5.0, -1.0, -2.0, 0.1)
log_q_drift = st.sidebar.slider("Log10 Q (Electrode Drift)", -7.0, -2.0, -5.0, 0.1)
log_r = st.sidebar.slider("Log10 R (Sensor Noise)", -3.0, 1.0, -1.0, 0.1)

fs = 1000
t, clean_sig = generate_wetware_signal(fs=fs, duration=1.5, spike_rate_hz=spike_rate, seed=seed)
p_weight = 0.0 if noise_mode == "AWGN Only" else 1.0
noisy_sig, true_drift = inject_wetware_noise(clean_sig, snr_db=snr_db, pink_weight=p_weight, drift_amp=drift_amp, seed=seed)

est_neural, est_drift, innovations, cov_trace = run_kalman_observer(
    noisy_sig, q_neural=10**log_q_neural, q_drift=10**log_q_drift, r_val=10**log_r
)

raw_rmse = calculate_rmse(clean_sig, noisy_sig)
twin_rmse = calculate_rmse(clean_sig, est_neural)
improvement = ((raw_rmse - twin_rmse) / raw_rmse) * 100 if raw_rmse > 0 else 0.0
innov_std = float(np.std(innovations))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input Wetware RMSE", f"{raw_rmse:.4f}")
c2.metric("Digital Twin RMSE", f"{twin_rmse:.4f}")
c3.metric("Error Reduction", f"{improvement:+.1f}%")
c4.metric("Innovation Residual (RMS)", f"{innov_std:.4f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 State Reconstruction",
    "📊 Spectral Band Analysis",
    "⚡ Spike & Action Potentials",
    "🩺 Twin Health & Diagnostics"
])

view = slice(0, 600)

with tab1:
    st.subheader("Time-Domain Neural Signal Reconstruction")
    render_plot(t[view], [
        ("Noisy Sensor Input", noisy_sig[view], "#4c78a8", 0.45, 1.0, "-"),
        ("Ground Truth Bio-Signal", clean_sig[view], "#72b7b2", 0.9, 1.3, "-"),
        ("Twin Neural Estimate", est_neural[view], "#ff6b6b", 0.95, 1.3, "--"),
    ])
    st.subheader("Isolated Electrode Baseline Wander (Drift State)")
    render_plot(t[view], [
        ("True Electrode Drift", true_drift[view], "#f58518", 0.85, 1.4, "-"),
        ("Estimated Baseline Drift", est_drift[view], "#54a24b", 0.95, 1.4, "--"),
    ], height=2.2)

with tab2:
    st.subheader("Power Spectral Density (Welch PSD)")
    f_raw, psd_raw = signal.welch(noisy_sig, fs, nperseg=256)
    f_clean, psd_clean = signal.welch(clean_sig, fs, nperseg=256)
    f_twin, psd_twin = signal.welch(est_neural, fs, nperseg=256)
    mask = f_raw <= 120
    render_plot(f_raw[mask], [
        ("Noisy Input PSD (dB)", 10 * np.log10(psd_raw[mask] + 1e-12), "#4c78a8", 0.55, 1.2, "-"),
        ("Ground Truth PSD (dB)", 10 * np.log10(psd_clean[mask] + 1e-12), "#72b7b2", 0.9, 1.4, "-"),
        ("Twin Estimate PSD (dB)", 10 * np.log10(psd_twin[mask] + 1e-12), "#ff6b6b", 0.95, 1.4, "--"),
    ], ylabel="PSD (dB/Hz)")

    bands = {"Delta (1-4 Hz)": (1, 4), "Theta (4-8 Hz)": (4, 8), "Alpha (8-13 Hz)": (8, 13), "Beta (13-30 Hz)": (13, 30), "Gamma (30-80 Hz)": (30, 80)}
    rows = []
    for name, (low, high) in bands.items():
        idx = (f_clean >= low) & (f_clean <= high)
        p_c = float(np.trapezoid(psd_clean[idx], f_clean[idx]))
        p_t = float(np.trapezoid(psd_twin[idx], f_twin[idx]))
        fid = max(0.0, 100.0 - abs(p_c - p_t) / (p_c + 1e-9) * 100.0)
        rows.append({"Band": name, "Ground Truth Power": f"{p_c:.4f}", "Twin Reconstructed Power": f"{p_t:.4f}", "Reconstruction Fidelity": f"{fid:.1f}%"})
    st.table(pd.DataFrame(rows))

with tab3:
    st.subheader("Extracellular Spike Detection (Adaptive Thresholding)")
    sigma = np.median(np.abs(est_neural)) / 0.6745
    thresh = -3.5 * sigma
    render_plot(t[view], [
        ("Twin Reconstructed Signal", est_neural[view], "#54a24b", 0.9, 1.2, "-"),
        ("Detection Threshold", np.full_like(t[view], thresh), "#ff4b4b", 0.85, 1.2, "--"),
    ])
    st.info(f"Calculated Noise Floor (σ): {sigma:.4f} | Spike Detection Threshold: {thresh:.4f} V")

with tab4:
    st.subheader("Kalman Innovation Residuals (Zero-Mean Whiteness Check)")
    render_plot(t[view], [("Innovation Residual", innovations[view], "#b279a2", 0.85, 1.1, "-")], height=2.2)
    st.subheader("State Error Covariance Trace (Filter Convergence)")
    render_plot(t[view], [("Trace(P) Convergence", cov_trace[view], "#eeca3b", 0.95, 1.4, "-")], height=2.2)
