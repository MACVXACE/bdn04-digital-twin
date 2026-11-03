import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from src.signal_generator import generate_wetware_baseline
from src.noise_injection import inject_wetware_noise
from src.twin_engine import WetwareDigitalTwin

st.set_page_config(page_title="BDN-04 Wetware Digital Twin", layout="wide", page_icon="🧠")

st.title("BDN-04 Biological Computing Digital Twin")
st.markdown("**Dual-State Observer & Real-Time Electrophysiology Reconstruction Engine**")

# Sidebar Controls
st.sidebar.header("Biological Wetware Config")
fs = 1000
duration = 6.0
spike_rate = st.sidebar.slider("Action Potential Firing Rate (Hz)", min_value=1.0, max_value=25.0, value=8.0, step=1.0)

st.sidebar.header("Electrode Noise & Artifacts")
noise_profile = st.sidebar.selectbox("Noise Profile", ["Pink (1/f) + Drift", "Pink (1/f)", "White (AWGN)"])
snr_db = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", min_value=-5.0, max_value=25.0, value=7.0, step=1.0)
drift_amp = st.sidebar.slider("Baseline Drift Amplitude", min_value=0.0, max_value=2.0, value=0.6, step=0.1)
seed = st.sidebar.number_input("RNG Seed", value=42, step=1)

st.sidebar.header("Twin Observer Tuning")
q_neural_log = st.sidebar.slider("Log10 Q (Neural Dynamics)", -5.0, -1.0, -2.0, 0.5)
q_drift_log = st.sidebar.slider("Log10 Q (Electrode Drift)", -7.0, -3.0, -5.0, 0.5)
r_sensor_log = st.sidebar.slider("Log10 R (Sensor Noise)", -4.0, 0.0, -2.0, 0.5)

# Run Pipeline
t, clean_lfp, spike_truth = generate_wetware_baseline(fs=fs, duration=duration, spike_rate_hz=spike_rate, rng_seed=int(seed))
noisy_signal = inject_wetware_noise(
    clean_lfp, 
    snr_db=snr_db, 
    drift_amp=drift_amp, 
    noise_type=noise_profile, 
    fs=fs, 
    rng_seed=int(seed)
)

twin = WetwareDigitalTwin(
    q_neural=10**q_neural_log, 
    q_drift=10**q_drift_log, 
    r_sensor=10**r_sensor_log
)
twin_out = twin.estimate(noisy_signal)
reconstructed = twin_out["neural_reconstruction"]
drift_est = twin_out["drift_estimate"]

# Top Metrics
noisy_rmse = np.sqrt(np.mean((clean_lfp - noisy_signal) ** 2))
twin_rmse = np.sqrt(np.mean((clean_lfp - reconstructed) ** 2))
error_reduction = ((noisy_rmse - twin_rmse) / noisy_rmse) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input Wetware RMSE", f"{noisy_rmse:.4f}")
c2.metric("Digital Twin RMSE", f"{twin_rmse:.4f}")
c3.metric("Error Reduction", f"{error_reduction:+.1f}%")
c4.metric("Innovation Residual (RMS)", f"{twin_out['residual_rms']:.4f}")

# Multi-Tab Dashboard
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 State Reconstruction", 
    "📊 Spectral Band Analysis", 
    "⚡ Spike & Action Potentials", 
    "🩺 Twin Health & Diagnostics"
])

view_window = 600  # First 0.6 seconds

with tab1:
    st.subheader("Time-Domain Neural Signal Reconstruction")
    main_df = pd.DataFrame({
        "Ground Truth Bio-Signal": clean_lfp[:view_window],
        "Noisy Sensor Input": noisy_signal[:view_window],
        "Twin Neural Estimate": reconstructed[:view_window]
    }, index=t[:view_window])
    st.line_chart(main_df)
    
    st.subheader("Isolated Electrode Baseline Wander (Drift State)")
    drift_df = pd.DataFrame({
        "Estimated Baseline Drift": drift_est[:view_window]
    }, index=t[:view_window])
    st.line_chart(drift_df)

with tab2:
    st.subheader("Welch Power Spectral Density (0 - 60 Hz)")
    f, psd_clean = signal.welch(clean_lfp, fs=fs, nperseg=512)
    _, psd_noisy = signal.welch(noisy_signal, fs=fs, nperseg=512)
    _, psd_twin = signal.welch(reconstructed, fs=fs, nperseg=512)
    
    mask = f <= 60.0
    psd_df = pd.DataFrame({
        "Ground Truth": 10 * np.log10(psd_clean[mask] + 1e-12),
        "Noisy Input": 10 * np.log10(psd_noisy[mask] + 1e-12),
        "Digital Twin": 10 * np.log10(psd_twin[mask] + 1e-12),
    }, index=f[mask])
    st.line_chart(psd_df)
    
    # Biological Frequency Bands
    bands = {
        "Delta (1 - 4 Hz)": (1, 4),
        "Theta (4 - 8 Hz)": (4, 8),
        "Alpha (8 - 12 Hz)": (8, 12),
        "Beta (12 - 30 Hz)": (12, 30),
        "Gamma (30 - 60 Hz)": (30, 60),
    }
    band_powers = []
    for band_name, (low, high) in bands.items():
        b_mask = (f >= low) & (f <= high)
        p_clean = np.trapezoid(psd_clean[b_mask], f[b_mask])
        p_twin = np.trapezoid(psd_twin[b_mask], f[b_mask])
        fidelity = max(0.0, 100.0 - abs((p_twin - p_clean) / (p_clean + 1e-9)) * 100.0)
        band_powers.append({"Band": band_name, "Truth Power": f"{p_clean:.3e}", "Twin Power": f"{p_twin:.3e}", "Fidelity": f"{fidelity:.1f}%"})
    st.dataframe(pd.DataFrame(band_powers))

with tab3:
    st.subheader("Extracellular Spike Detection (Adaptive Thresholding)")
    # Spike detection using 3.5x median-based standard deviation estimator
    sigma_n = np.median(np.abs(reconstructed)) / 0.6745
    threshold = -3.5 * sigma_n
    detected_spikes = (reconstructed < threshold).astype(int)
    
    spike_df = pd.DataFrame({
        "Twin Reconstructed Signal": reconstructed[:view_window],
        "Detection Threshold": [threshold] * view_window
    }, index=t[:view_window])
    st.line_chart(spike_df)
    st.info(f"Calculated Noise Floor (σ): {sigma_n:.4f} | Spike Detection Threshold: {threshold:.4f} V")

with tab4:
    st.subheader("Kalman Filter Innovation Sequence (Residual Diagnostics)")
    st.markdown("The innovation residual $\\tilde{y}_k = z_k - \\mathbf{H}\\hat{\\mathbf{x}}_k$ represents the unexplained physical component. Zero-mean white innovation confirms optimal observer tracking.")
    res_df = pd.DataFrame({
        "Innovation Residual": twin_out["innovations"][:view_window]
    }, index=t[:view_window])
    st.line_chart(res_df)
