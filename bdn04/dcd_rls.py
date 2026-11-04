"""
BDN-04 :: Recovery layer, DCD-RLS adaptive noise canceller
==========================================================

Exponentially weighted RLS where the normal equations are solved incrementally
by leading-element Dichotomous Coordinate Descent:

    Zakharov, White & Liu (2008), "Low-complexity RLS algorithms using
    dichotomous coordinate descent iterations", IEEE Trans. Signal Process.
    56(7):3150-3161.

Per sample n:
    R(n)   = lam R(n-1) + x(n) x(n)^T      (O(N) via time-shift structure, transversal filter)
    e(n)   = d(n) - w(n-1)^T x(n)          (a priori error = ANC output)
    beta0  = lam r(n-1) + e(n) x(n)
    solve R(n) dw = beta0 with <= Nu DCD updates -> dw, r(n)
    w(n)   = w(n-1) + dw

The DCD step size alpha is always a power of two (H/2, H/4, ... H/2^Mb), so in
fixed-point hardware each update is a bit shift plus N additions, with no
multiplications and no divisions. This is the edge-silicon argument: about O(N)
multiplications per sample for the R update and the filter output, against
O(N^2) for classical RLS, with near-RLS convergence on coloured (high
eigenvalue-spread) inputs like sEMG, where NLMS stalls.

Topology (Widrow ANC):
    d(n) = s(n) + v(n)       primary: data + biological interference
    u(n) -> v(n) via unknown tissue/electrode coupling path
    u(n)                     reference electrode/channel: interference, no data
    e(n) -> s_hat(n)         cleaned output
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pure-Python fallback, same code
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(fn):
            return fn
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True, fastmath=False)
def _dcd_rls_kernel(u, d, N, lam, delta, H, Mb, Nu, w_stride):
    T = d.shape[0]
    w = np.zeros(N)
    r = np.zeros(N)
    x = np.zeros(N)
    R = np.eye(N) * delta
    e = np.empty(T)
    y = np.empty(T)
    n_upd = np.empty(T, dtype=np.int32)
    n_snap = (T + w_stride - 1) // w_stride
    W = np.empty((n_snap, N))

    for n in range(T):
        # regressor shift
        for i in range(N - 1, 0, -1):
            x[i] = x[i - 1]
        x[0] = u[n]

        # R update exploiting time-shift: R_n[1:,1:] = R_{n-1}[:-1,:-1]
        for i in range(N - 1, 0, -1):
            for j in range(N - 1, 0, -1):
                R[i, j] = R[i - 1, j - 1]
        for j in range(N):
            v = lam * R[0, j] + x[0] * x[j]
            R[0, j] = v
            R[j, 0] = v

        # a priori output and error
        yn = 0.0
        for i in range(N):
            yn += w[i] * x[i]
        en = d[n] - yn
        y[n] = yn
        e[n] = en

        # beta0 = lam r + e x  (stored in r)
        for i in range(N):
            r[i] = lam * r[i] + en * x[i]

        # leading-element DCD: solve R dw = r, applying dw to w directly
        alpha = H * 0.5
        m = 1
        k = 0
        stop = False
        while k < Nu and not stop:
            p = 0
            rmax = abs(r[0])
            for i in range(1, N):
                ai = abs(r[i])
                if ai > rmax:
                    rmax = ai
                    p = i
            while rmax <= 0.5 * alpha * R[p, p]:
                m += 1
                alpha *= 0.5
                if m > Mb:
                    stop = True
                    break
            if stop:
                break
            s = alpha if r[p] > 0 else -alpha
            w[p] += s
            for i in range(N):
                r[i] -= s * R[i, p]
            k += 1
        n_upd[n] = k

        if n % w_stride == 0:
            W[n // w_stride, :] = w

    return y, e, w, W, n_upd


class DCDRLS:
    """DCD-RLS adaptive transversal filter.

    n_taps : filter order N (coupling-path memory)
    lam    : forgetting factor; effective memory ~ 1/(1-lam) samples
    delta  : R(0) = delta * I regularisation
    H      : DCD amplitude range (upper bound on |dw| per sample; power of two)
    Mb     : bits of DCD precision (smallest step H / 2^Mb)
    Nu     : max DCD updates per sample (complexity knob; Nu ~ 1-8 is typical)
    """

    def __init__(self, n_taps=16, lam=0.995, delta=1e-2, H=1.0, Mb=16, Nu=4):
        self.N, self.lam, self.delta, self.H, self.Mb, self.Nu = n_taps, lam, delta, H, Mb, Nu

    def run(self, u, d, w_stride=1):
        u = np.ascontiguousarray(u, dtype=np.float64)
        d = np.ascontiguousarray(d, dtype=np.float64)
        y, e, w, W, n_upd = _dcd_rls_kernel(u, d, self.N, self.lam, self.delta,
                                            self.H, self.Mb, self.Nu, w_stride)
        return {"y": y, "e": e, "w": w, "W": W, "n_updates": n_upd, "w_stride": w_stride}


# --------------------------------------------------------------------------- #
# Baselines for the side-by-side (exact RLS = upper bound, NLMS = what people ship)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _rls_kernel(u, d, N, lam, delta):
    T = d.shape[0]
    w = np.zeros(N)
    P = np.eye(N) / delta
    x = np.zeros(N)
    e = np.empty(T)
    for n in range(T):
        for i in range(N - 1, 0, -1):
            x[i] = x[i - 1]
        x[0] = u[n]
        Px = P @ x
        k = Px / (lam + x @ Px)
        en = d[n] - w @ x
        e[n] = en
        w += k * en
        P = (P - np.outer(k, Px)) / lam
    return e, w


@njit(cache=True)
def _nlms_kernel(u, d, N, mu, eps):
    T = d.shape[0]
    w = np.zeros(N)
    x = np.zeros(N)
    e = np.empty(T)
    for n in range(T):
        for i in range(N - 1, 0, -1):
            x[i] = x[i - 1]
        x[0] = u[n]
        en = d[n] - w @ x
        e[n] = en
        w += (mu / (eps + x @ x)) * en * x
    return e, w


def rls(u, d, n_taps=16, lam=0.995, delta=1e-2):
    return _rls_kernel(np.asarray(u, float), np.asarray(d, float), n_taps, lam, delta)


def nlms(u, d, n_taps=16, mu=0.5, eps=1e-6):
    return _nlms_kernel(np.asarray(u, float), np.asarray(d, float), n_taps, mu, eps)


def misalignment_db(W: np.ndarray, h_true: np.ndarray) -> np.ndarray:
    """Normalised weight misalignment 10log10(||w - h||^2 / ||h||^2) per snapshot."""
    h = np.zeros(W.shape[1])
    h[:min(h.size, h_true.size)] = h_true[:h.size]
    return 10 * np.log10(np.sum((W - h) ** 2, axis=1) / np.sum(h ** 2))
