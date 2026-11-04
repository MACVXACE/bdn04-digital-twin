import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from src.signal_generator import generate_wetware_signal, inject_wetware_noise
from src.twin_engine import run_kalman_observer, calculate_rmse

st.set_page_config(page_title="BDN-04 Wetware Digital Twin", layout="wide")

def svg_chart(x, series, height=220, y_label=""):
    w, h = 900, height
    pad_l, pad_r, pad_t, pad_b = 55, 20, 25, 30
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    all_y = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if abs(ymax - ymin) < 1e-9:
        ymin -= 1.0
        ymax += 1.0
    xmin, xmax = float(x[0]), float(x[-1])
    if abs(xmax - xmin) < 1e-9:
        xmax += 1.0

    lines = [f"<svg viewBox=\"0 0 {w} {h}\" style=\"width:100%;height:auto;background:#0e1117;border:1px solid #262730;border-radius:8px;margin-bottom:12px;font-family:sans-serif;\">"]
    for i in range(5):
        gy = pad_t + ph * (i / 4.0)
        gval = ymax - (ymax - ymin) * (i / 4.0)
        lines.append(f"<line x1=\"{pad_l}\" y1=\"{gy:.1f}\" x2=\"{w-pad_r}\" y2=\"{gy:.1f}\" stroke=\"#262730\" stroke-width=\"1\"/>")
        lines.append(f"<text x=\"{pad_l-8}\" y=\"{gy+4:.1f}\" fill=\"#a3a8b8\" font-size=\"11\" text-anchor=\"end\">{gval:.2f}</text>")

    for i in range(6):
        gx = pad_l + pw * (i / 5.0)
        xval = xmin + (xmax - xmin) * (i / 5.0)
        lines.append(f"<text x=\"{gx:.1f}\" y=\"{h-8}\" fill=\"#a3a8b8\" font-size=\"11\" text-anchor=\"middle\">{xval:.2f}</text>")

    for idx, (label, y, color, dash) in enumerate(series):
        y_arr = np.asarray(y, dtype=float)
        xs = pad_l + (x - xmin) / (xmax - xmin) * pw
        ys = pad_t + (ymax - y_arr) / (ymax - ymin) * ph
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(xs, ys))
        da = f"stroke-dasharray=\"{dash}\"" if dash else ""
        lines.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.8\" {da} points=\"{pts}\"/>")
        lx = pad_l + 10 + idx * 190
        lines.append(f"<rect x=\"{lx}\" y=\"6\" width=\"12\" height=\"12\" fill=\"{color}\" rx=\"2\"/>")
        lines.append(f"<text x=\"{lx+18}\" y=\"16\" fill=\"#fafafa\" font-size=\"12\">{label}</text>")

    lines.append("</svg>")
    st.markdown("".join(lines), unsafe_allow_html=True)

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
    svg_chart(t[view], [
        ("Noisy Sensor Input", noisy_sig[view], "#4c78a8", ""),
        ("Ground Truth Bio-Signal", clean_sig[view], "#72b7b2", ""),
        ("Twin Neural Estimate", est_neural[view], "#ff6b6b", "5,3"),
    ])
    st.subheader("Isolated Electrode Baseline Wander (Drift State)")
    svg_chart(t[view], [
        ("True Electrode Drift", true_drift[view], "#f58518", ""),
        ("Estimated Baseline Drift", est_drift[view], "#54a24b", "5,3"),
    ], height=190)

with tab2:
    st.subheader("Power Spectral Density (Welch PSD)")
    f_raw, psd_raw = signal.welch(noisy_sig, fs, nperseg=256)
    f_clean, psd_clean = signal.welch(clean_sig, fs, nperseg=256)
    f_twin, psd_twin = signal.welch(est_neural, fs, nperseg=256)
    mask = f_raw <= 120
    svg_chart(f_raw[mask], [
        ("Noisy Input PSD (dB)", 10 * np.log10(psd_raw[mask] + 1e-12), "#4c78a8", ""),
        ("Ground Truth PSD (dB)", 10 * np.log10(psd_clean[mask] + 1e-12), "#72b7b2", ""),
        ("Twin Estimate PSD (dB)", 10 * np.log10(psd_twin[mask] + 1e-12), "#ff6b6b", "5,3"),
    ])

    bands = {"Delta (1-4 Hz)": (1, 4), "Theta (4-8 Hz)": (4, 8), "Alpha (8-13 Hz)": (8, 13), "Beta (13-30 Hz)": (13, 30), "Gamma (30-80 Hz)": (30, 80)}
    md_table = "| Spectral Band | Ground Truth Power | Twin Reconstructed Power | Reconstruction Fidelity |\n|---|---|---|---|\n"
    for name, (low, high) in bands.items():
        idx = (f_clean >= low) & (f_clean <= high)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        p_c = float(trapz_fn(psd_clean[idx], f_clean[idx]))
        p_t = float(trapz_fn(psd_twin[idx], f_twin[idx]))
        fid = max(0.0, 100.0 - abs(p_c - p_t) / (p_c + 1e-9) * 100.0)
        md_table += f"| {name} | {p_c:.4f} | {p_t:.4f} | {fid:.1f}% |\n"
    st.markdown(md_table)

with tab3:
    st.subheader("Extracellular Spike Detection (Adaptive Thresholding)")
    sigma = np.median(np.abs(est_neural)) / 0.6745
    thresh = -3.5 * sigma
    svg_chart(t[view], [
        ("Twin Reconstructed Signal", est_neural[view], "#54a24b", ""),
        ("Detection Threshold", np.full_like(t[view], thresh), "#ff4b4b", "6,4"),
    ])
    st.info(f"Calculated Noise Floor (σ): {sigma:.4f} | Spike Detection Threshold: {thresh:.4f} V")

with tab4:
    st.subheader("Kalman Innovation Residuals (Zero-Mean Whiteness Check)")
    svg_chart(t[view], [("Innovation Residual", innovations[view], "#b279a2", "")], height=190)
    st.subheader("State Error Covariance Trace (Filter Convergence)")
    svg_chart(t[view], [("Trace(P) Convergence", cov_trace[view], "#eeca3b", "")], height=190)
