# BDN-04 Wetware Computing Digital Twin

[![CI Tests](https://github.com/MACVXACE/bdn04-digital-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/MACVXACE/bdn04-digital-twin/actions)

Real-time state-space digital twin and 2-state Kalman observer for wetware biological neural network electrophysiology.

## 🚀 **Live Interactive App:** [macvxace-bdn04-digital-twin.streamlit.app](https://macvxace-bdn04-digital-twin-app-gcxojn.streamlit.app)

Overview
- **Decoupled 2-State Kalman Filter**: Separates biological LFPs and action potential spikes from electrode baseline wander.
- **Biophysical Signal Engine**: Generates composite Theta/Alpha/Gamma oscillations with stochastic Poisson spike trains.
- **Noise Synthesis**: Simulates real MEA wetware environments with 1/f pink noise and slow electrochemical drift.
- **Telemetry & Diagnostics**: Online innovation residual tracking, Welch PSD spectral band decomposition, and adaptive spike detection.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
streamlit run app.py
```

## Automated Testing
```bash
python -m pytest -v
```
