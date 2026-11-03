import numpy as np

def generate_wetware_baseline(
    fs: int = 1000, 
    duration: float = 10.0, 
    spike_rate_hz: float = 8.0, 
    rng_seed: int = 42
) -> tuple:
    """
    Synthesizes biophysically realistic wetware electrophysiology:
    1. Composite LFP oscillations (Theta 6 Hz, Alpha 10 Hz, Gamma 40 Hz).
    2. Stochastic extracellular neural action potentials (spikes).
    """
    rng = np.random.default_rng(rng_seed)
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # 1. Local Field Potentials (LFP)
    theta = 0.8 * np.sin(2 * np.pi * 6.0 * t)
    alpha = 0.4 * np.sin(2 * np.pi * 10.0 * t + 0.4)
    gamma = 0.15 * np.sin(2 * np.pi * 40.0 * t + 1.2)
    lfp = theta + alpha + gamma
    
    # 2. Extracellular Action Potential Spikes (Biphasic waveforms)
    spike_train = np.zeros(n_samples)
    dt = 1.0 / fs
    n_spikes = rng.poisson(spike_rate_hz * duration)
    spike_times = rng.uniform(0.05, duration - 0.05, size=n_spikes)
    
    # Standard 2 ms biphasic extracellular spike template
    spike_len = int(0.003 * fs)  # 3 ms template
    if spike_len % 2 == 0:
        spike_len += 1
    t_spike = np.linspace(-1.5, 1.5, spike_len)
    spike_kernel = -2.5 * np.exp(-0.5 * (t_spike / 0.4)**2) + 1.2 * np.exp(-0.5 * ((t_spike - 0.7) / 0.5)**2)
    
    for st in spike_times:
        idx = int(st * fs)
        start = idx - len(spike_kernel) // 2
        end = start + len(spike_kernel)
        if start >= 0 and end < n_samples:
            spike_train[start:end] += spike_kernel
            
    clean_signal = lfp + spike_train
    return t, clean_signal, spike_train
