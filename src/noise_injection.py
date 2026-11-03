import numpy as np

def generate_pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generates 1/f pink noise via spectral shaping of white Gaussian noise."""
    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # Avoid division by zero at DC
    
    # Scale spectral amplitudes by 1/sqrt(f) to get 1/f power spectrum
    s_pink = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(s_pink, n=n_samples)
    return (pink - np.mean(pink)) / np.std(pink)

def inject_wetware_noise(
    clean_signal: np.ndarray, 
    snr_db: float, 
    drift_amp: float = 0.3,
    drift_freq: float = 0.2,
    noise_type: str = "Pink (1/f) + Drift",
    fs: int = 1000,
    rng_seed: int = 42
) -> np.ndarray:
    """Injects biological wetware noise: AWGN, 1/f pink noise, and electrode drift."""
    rng = np.random.default_rng(rng_seed)
    n = len(clean_signal)
    t = np.linspace(0, n / fs, n, endpoint=False)
    
    if noise_type == "White (AWGN)":
        raw_noise = rng.standard_normal(n)
    elif noise_type == "Pink (1/f)":
        raw_noise = generate_pink_noise(n, rng)
    else:  # "Pink (1/f) + Drift"
        pink = generate_pink_noise(n, rng)
        # Slow sinusoidal/polynomial baseline wander
        wander = np.sin(2 * np.pi * drift_freq * t) + 0.3 * np.cos(2 * np.pi * 0.05 * t)
        raw_noise = pink + drift_amp * wander

    # Scale noise to target SNR relative to clean signal power
    sig_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(raw_noise ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    scaled_noise = raw_noise * np.sqrt(sig_power / (snr_linear * noise_power))
    
    return clean_signal + scaled_noise
