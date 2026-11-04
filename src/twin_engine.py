import numpy as np

def run_kalman_observer(noisy_sig, q_neural=1e-2, q_drift=1e-5, r_val=1e-1):
    n = len(noisy_sig)
    A = np.array([[0.98, 0.0], [0.0, 0.9995]])
    H = np.array([[1.0, 1.0]])
    Q = np.array([[q_neural, 0.0], [0.0, q_drift]])
    R = float(r_val)
    x = np.zeros((2, 1))
    P = np.eye(2) * 0.5
    est_neural = np.zeros(n)
    est_drift = np.zeros(n)
    innovations = np.zeros(n)
    cov_trace = np.zeros(n)
    for k in range(n):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        z = noisy_sig[k]
        y_tilde = z - (H @ x_pred).item()
        S = (H @ P_pred @ H.T).item() + R
        K = (P_pred @ H.T) / S
        x = x_pred + K * y_tilde
        P = (np.eye(2) - K @ H) @ P_pred
        est_neural[k] = x[0, 0]
        est_drift[k] = x[1, 0]
        innovations[k] = y_tilde
        cov_trace[k] = np.trace(P)
    return est_neural, est_drift, innovations, cov_trace

def calculate_rmse(true_sig, est_sig):
    return float(np.sqrt(np.mean((np.asarray(true_sig) - np.asarray(est_sig)) ** 2)))
