import streamlit as st
import numpy as np
import pandas as pd
from src.signal_generator import generate_baseline
from src.noise_injection import inject_awgn
from src.twin_engine import DigitalTwinEstimator

st.set_page_config(page_title="BDN04 Digital Twin", layout="wide")

st.title("BDN04 Digital Twin: Signal State Reconstruction")
st.markdown("Live comparison of Ground Truth Baseline, Noise-Injected Wetware Signal, and Digital Twin State Estimation.")

# Sidebar controls
st.sidebar.header("Signal & Noise Parameters")
snr_db = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", min_value=-10.0, max_value=30.0, value=8.0, step=1.0)
seed = st.sidebar.number_input("RNG Seed", value=42, step=1)

st.sidebar.header("Digital Twin Estimator")
q_log = st.sidebar.slider("Log10 Process Variance (Q)", min_value=-6.0, max_value=-1.0, value=-3.0, step=0.5)
r_log = st.sidebar.slider("Log10 Measurement Variance (R)", min_value=-4.0, max_value=1.0, value=-1.0, step=0.5)

# Generate baseline and inject AWGN
t, clean_signal = generate_baseline()
noisy_signal = inject_awgn(clean_signal, snr_db=snr_db, rng_seed=seed)

# Run digital twin estimation
twin = DigitalTwinEstimator(process_variance=10**q_log, measurement_variance=10**r_log)
twin_reconstruction = twin.estimate_state(noisy_signal)

# Performance Metrics
noisy_rmse = np.sqrt(np.mean((clean_signal - noisy_signal) ** 2))
twin_rmse = np.sqrt(np.mean((clean_signal - twin_reconstruction) ** 2))
improvement = ((noisy_rmse - twin_rmse) / noisy_rmse) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Input Signal RMSE", f"{noisy_rmse:.4f}")
col2.metric("Digital Twin RMSE", f"{twin_rmse:.4f}")
col3.metric("Error Reduction", f"{improvement:+.1f}%")

# Plot first 500 samples (0.5 s window)
plot_df = pd.DataFrame({
    "Clean Baseline (Ground Truth)": clean_signal[:500],
    "Noisy Wetware Input": noisy_signal[:500],
    "Digital Twin Estimate": twin_reconstruction[:500]
}, index=t[:500])

st.subheader("Signal Trajectory")
st.line_chart(plot_df)
