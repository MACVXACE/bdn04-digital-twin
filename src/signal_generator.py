import numpy as np
from scipy import signal

def generate_baseline(fs: int = 1000, duration: float = 10.0) -> tuple:
    """Synthesizes a clean reference signal (sine + sawtooth wave)."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * signal.sawtooth(2 * np.pi * 1.2 * t)
    return t, clean_signal
