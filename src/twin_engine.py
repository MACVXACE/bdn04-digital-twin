import numpy as np

class WetwareDigitalTwin:
    """
    2-State Discrete Kalman Observer for Wetware Neural Interfaces:
    - State x_0: Neural state (oscillations + fast dynamics)
    - State x_1: Electrode baseline wander / impedance drift
    """
    def __init__(self, q_neural: float = 1e-2, q_drift: float = 1e-5, r_sensor: float = 1e-2):
        self.q_neural = q_neural
        self.q_drift = q_drift
        self.r_sensor = r_sensor
        
    def estimate(self, noisy_signal: np.ndarray) -> dict:
        n = len(noisy_signal)
        
        # State transition matrix A (identity with slight decay on neural state)
        A = np.array([[0.995, 0.0],
                      [0.0,   1.0]])
        
        # Process noise covariance Q
        Q = np.array([[self.q_neural, 0.0],
                      [0.0,           self.q_drift]])
        
        # Measurement matrix H: sensor measures sum of neural + drift
        H = np.array([[1.0, 1.0]])
        R = self.r_sensor
        
        # Allocation arrays
        x_neural_est = np.zeros(n)
        x_drift_est = np.zeros(n)
        innovations = np.zeros(n)
        
        # Initial priors
        x = np.array([noisy_signal[0], 0.0])
        P = np.diag([1.0, 1.0])
        
        for k in range(n):
            # 1. Predict
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            
            # 2. Measurement update
            z = noisy_signal[k]
            y_tilde = z - float(H @ x_pred)  # Innovation / residual
            S = float(H @ P_pred @ H.T) + R   # Innovation covariance
            K = (P_pred @ H.T) / S            # Optimal Kalman Gain (2x1)
            
            x = x_pred + K.flatten() * y_tilde
            P = (np.eye(2) - K @ H) @ P_pred
            
            # Record states
            x_neural_est[k] = x[0]
            x_drift_est[k] = x[1]
            innovations[k] = y_tilde
            
        return {
            "neural_reconstruction": x_neural_est,
            "drift_estimate": x_drift_est,
            "innovations": innovations,
            "residual_rms": np.sqrt(np.mean(innovations**2))
        }
