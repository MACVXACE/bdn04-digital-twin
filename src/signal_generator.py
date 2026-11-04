import numpy as np

def generate_wetware_signal(fs=1000, duration=1.5, spike_rate_hz=8.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    lfp = (
        0.45 * np.sin(2 * np.pi * 6.0 * t)
        + 0.30 * np.sin(2 * np.pi * 10.0 * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * 40.0 * t + 1.2)
    )
    spikes = np.zeros_like(t)
    n_spikes = rng.poisson(spike_rate_hz * duration)
    if n_spikes > 0:
        locs = rng.choice(len(t) - 20, size=n_spikes, replace=False)
        k_t = np.arange(20) / fs
        waveform = -1.8 * np.exp(-k_t / 0.0015) * np.sin(2 * np.pi * 500 * k_t)
        for idx in locs:
            spikes[idx:idx+20] += waveform
    return t, lfp + spikes

def inject_wetware_noise(clean_sig, snr_db=7.0, pink_weight=1.0, drift_amp=0.6, seed=42):
    rng = np.random.default_rng(seed)
    n = len(clean_sig)
    white = rng.standard_normal(n)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(fft_vals * scale, n=n)
    pink = pink / (np.std(pink) + 1e-12)
    raw_noise = pink_weight * pink + (1.0 - pink_weight) * white
    raw_noise = raw_noise / (np.std(raw_noise) + 1e-12)
    sig_power = np.mean(clean_sig ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scaled_noise = raw_noise * np.sqrt(target_noise_power)
    t_norm = np.linspace(0, 1, n)
    drift = drift_amp * (0.6 * np.sin(2 * np.pi * 0.7 * t_norm) + 0.4 * np.cos(2 * np.pi * 1.8 * t_norm))
    return clean_sig + scaled_noise + drift, drift
