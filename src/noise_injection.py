import numpy as np

def inject_awgn(clean_signal: np.ndarray, snr_db: float, rng_seed: int = 42) -> np.ndarray:
    """Injects additive white Gaussian noise (AWGN) at a target SNR (dB)."""
    rng = np.random.default_rng(rng_seed)
    sig_power = np.mean(clean_signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=clean_signal.shape)
    return clean_signal + noise
