import numpy as np
import pytest
from src.signal_generator import generate_wetware_baseline
from src.noise_injection import inject_wetware_noise
from src.twin_engine import WetwareDigitalTwin

def test_signal_generator_shape():
    fs, duration = 1000, 2.0
    t, clean, spikes = generate_wetware_baseline(fs=fs, duration=duration)
    assert len(t) == int(fs * duration)
    assert len(clean) == int(fs * duration)
    assert len(spikes) == int(fs * duration)

def test_noise_injection_snr():
    fs = 1000
    t, clean, _ = generate_wetware_baseline(fs=fs, duration=2.0)
    noisy = inject_wetware_noise(clean, snr_db=10.0, noise_type="Pink (1/f) + Drift", fs=fs)
    assert len(noisy) == len(clean)
    assert not np.isnan(noisy).any()

def test_twin_observer_error_reduction():
    fs = 1000
    t, clean, _ = generate_wetware_baseline(fs=fs, duration=3.0, rng_seed=42)
    noisy = inject_wetware_noise(clean, snr_db=5.0, drift_amp=0.5, fs=fs, rng_seed=42)
    
    twin = WetwareDigitalTwin(q_neural=0.1, q_drift=1e-5, r_sensor=1.0)
    out = twin.estimate(noisy)
    
    reconstructed = out["neural_reconstruction"]
    noisy_rmse = np.sqrt(np.mean((clean - noisy) ** 2))
    twin_rmse = np.sqrt(np.mean((clean - reconstructed) ** 2))
    
    # Observer must outperform raw sensor noise
    assert twin_rmse < noisy_rmse
