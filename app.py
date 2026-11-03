import streamlit as st
import pandas as pd
from src.signal_generator import generate_baseline
from src.noise_injection import inject_awgn

st.set_page_config(page_title="Digital Twin", layout="wide")

st.title("BDN04 Digital Twin: Signal Noise Injector")
st.markdown("Interactive baseline synthesis and AWGN injection.")

# Sidebar controls
st.sidebar.header("Signal Parameters")
snr_db = st.sidebar.slider("Signal-to-Noise Ratio (SNR)", min_value=-10.0, max_value=30.0, value=10.0, step=1.0)

# Generate and process data
t, clean_signal = generate_baseline()
noisy_signal = inject_awgn(clean_signal, snr_db)

# Prepare first 500 samples (0.5 seconds) for clean visualization
df = pd.DataFrame({
    "Clean Baseline": clean_signal[:500],
    "Noisy Signal": noisy_signal[:500]
}, index=t[:500])

# Render UI
st.subheader(f"Signal Comparison (SNR: {snr_db} dB)")
st.line_chart(df)
