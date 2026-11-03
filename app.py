import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from src.signal_generator import generate_baseline
from src.noise_injection import inject_wetware_noise
from src.twin_engine import DigitalTwinEstimator

st.set_page_config(page_title="BDN04 Digital Twin", layout="wide")

st.title("BDN04 Digital Twin: Signal State Reconstruction")
st.markdown("Live comparison of Ground Truth Baseline, Noise-Injected Wetware Signal, and Digital Twin State Estimation.")

# Sidebar controls
st.sidebar.header("Signal & Wetware Noise")
noise_profile = st.sidebar.selectbox(
    "Noise Profile", 
    ["Pink (1/f) + Drift", "Pink (1/f)", "White (AWGN)"]
)
snr_db = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", min_value=-10.0, max_value=30.0, value=8.0, step=1.0)
drift_amp = st.sidebar.slider("Baseline Drift Amplitude", min_value=0.0, max_value=1.5, value=0.4, step=0.1)
seed = st.sidebar.number_input("RNG Seed", value=42, step=1)

st.sidebar.header("Digital Twin Estimator")
q_log = st.sidebar.slider("Log10 Process Variance (Q)", min_value=-6.0, max_value=-1.0, value=-3.0, step=0.5)
r_log = st.sidebar.slider("Log10 Measurement Variance (R)", min_value=-4.0, max_value=1.0, value=-1.0, step=0.5)

# 1. Synthesize baseline & inject realistic wetware noise
fs = 1000
t, clean_signal = generate_baseline(fs=fs, duration=10.0)
noisy_signal = inject_wetware_noise(
    clean_signal, 
    snr_db=snr_db, 
    drift_amp=drift_amp, 
    noise_type=noise_profile, 
    fs=fs, 
    rng_seed=int(seed)
)

# 2. Digital Twin Reconstruction
twin = DigitalTwinEstimator(process_variance=10**q_log, measurement_variance=10**r_log)
twin_reconstruction = twin.estimate_state(noisy_signal)

# 3. Performance Metrics
noisy_rmse = np.sqrt(np.mean((clean_signal - noisy_signal) ** 2))
twin_rmse = np.sqrt(np.mean((clean_signal - twin_reconstruction) ** 2))
improvement = ((noisy_rmse - twin_rmse) / noisy_rmse) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Input Signal RMSE", f"{noisy_rmse:.4f}")
col2.metric("Digital Twin RMSE", f"{twin_rmse:.4f}")
col3.metric("Error Reduction", f"{improvement:+.1f}%")

# 4. Time Domain Trajectory
plot_df = pd.DataFrame({
    "Clean Baseline (Ground Truth)": clean_signal[:500],
    "Noisy Wetware Input": noisy_signal[:500],
    "Digital Twin Estimate": twin_reconstruction[:500]
}, index=t[:500])

st.subheader("Time Domain Trajectory (First 0.5s)")
st.line_chart(plot_df)

# 5. Frequency Domain Analysis (Welch PSD)
st.subheader("Frequency Domain: Power Spectral Density (PSD)")
f_clean, psd_clean = signal.welch(clean_signal, fs=fs, nperseg=512)
f_noisy, psd_noisy = signal.welch(noisy_signal, fs=fs, nperseg=512)
f_twin, psd_twin = signal.welch(twin_reconstruction, fs=fs, nperseg=512)

# Limit plot to 0 - 60 Hz where primary signal and drift sit
mask = f_clean <= 60.0
psd_df = pd.DataFrame({
    "Clean Baseline": 10 * np.log10(psd_clean[mask] + 1e-12),
    "Noisy Wetware": 10 * np.log10(psd_noisy[mask] + 1e-12),
    "Digital Twin": 10 * np.log10(psd_twin[mask] + 1e-12),
}, index=f_clean[mask])

st.line_chart(psd_df)
