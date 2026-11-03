# BDN-04 Wetware Computing Digital Twin

A real-time state-space digital twin and signal reconstruction engine for biological neural networks (wetware computing / multi-electrode array electrophysiology).

## System Architecture
### State-Space Representation
The twin models the electrophysiological environment as a discrete two-state stochastic system:

$$
\mathbf{x}_{k} = \begin{bmatrix} x_{\text{neural}} \\ x_{\text{drift}} \end{bmatrix}_{k} = \mathbf{A} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \quad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q})
$$

$$
z_k = \mathbf{H} \mathbf{x}_k + v_k, \quad \mathbf{H} = [1, 1], \quad v_k \sim \mathcal{N}(0, R)
$$

This formulation enables live decoupling of high-frequency bio-oscillations from low-frequency electrode polarization wander.

## Features
- **Biophysical Synthesis**: Composite Local Field Potentials ($\theta, \alpha, \gamma$) with Poisson extracellular action potential bursts.
- **Wetware Noise Profile**: Realistic $1/f$ pink flicker noise and slow electrode baseline drift.
- **2-State Observer**: Decoupled state estimation isolating neural trajectories from impedance drift.
- **Spectral Band Decomposition**: Real-time Welch PSD tracking power fidelity across $\delta, \theta, \alpha, \beta, \gamma$ bands.
- **Zero-Bloat Data Hygiene**: Strict `.gitignore` policy preventing large array files from exceeding GitHub push limits.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
