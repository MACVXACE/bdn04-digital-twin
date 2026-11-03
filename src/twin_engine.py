import numpy as np

class DigitalTwinEstimator:
    """1D Kalman filter state estimator representing the digital twin dynamics."""
    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 1e-2):
        self.q = process_variance  # Process noise covariance (Q)
        self.r = measurement_variance  # Measurement noise covariance (R)
        
    def estimate_state(self, noisy_signal: np.ndarray) -> np.ndarray:
        """Filters noisy observation to reconstruct the underlying physical state."""
        n = len(noisy_signal)
        estimated_signal = np.zeros(n)
        
        # Initial state and error covariance
        x_hat = noisy_signal[0]
        p = 1.0
        
        for k in range(n):
            # Time update (Predict)
            p = p + self.q
            
            # Measurement update (Correct)
            kalman_gain = p / (p + self.r)
            x_hat = x_hat + kalman_gain * (noisy_signal[k] - x_hat)
            p = (1.0 - kalman_gain) * p
            
            estimated_signal[k] = x_hat
            
        return estimated_signal
