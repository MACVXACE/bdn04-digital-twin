"""Validation harness. Run: python -m bdn04.selftest"""
import time
import numpy as np
from scipy.signal import lfilter

from bdn04.tissue import (TISSUES, ChannelConfig, eqs_channel, eqs_channel_analytic, path_tissue_impedance,
                          channel_interpolator, apply_channel)
from bdn04.dcd_rls import DCDRLS, rls, nlms, misalignment_db

# ---------------------------------------------------------------- 1. dielectrics
# IT'IS Foundation tissue DB v4.x (same Gabriel parametrisation), sigma [S/m], eps_r
REF = {("muscle", 1e5): (0.362, 8089), ("muscle", 1e6): (0.503, 1836)}
print("== Cole-Cole vs IT'IS reference ==")
for (t, f), (s_ref, e_ref) in REF.items():
    s, e = TISSUES[t].sigma(f)[0], TISSUES[t].eps_r(f)[0]
    print(f"{t:7s} {f/1e3:7.0f} kHz  sigma={s:.4f} (ref {s_ref})  eps_r={e:8.1f} (ref {e_ref})"
          f"  err {100*(s-s_ref)/s_ref:+.1f}% / {100*(e-e_ref)/e_ref:+.1f}%")
print("@500 kHz:", {k: (round(v.sigma(5e5)[0], 4), round(v.eps_r(5e5)[0], 1)) for k, v in TISSUES.items()})

# ---------------------------------------------------------------- 2. channel
f = np.logspace(3, 7, 240)
fc = 5e5
i_fc = np.argmin(abs(f - fc))
print("\n== EQS-HBC channel |H| @ 500 kHz ==")
for path, ds in (("wrist_to_head", (0.15, 0.5, 0.95)), ("wrist_to_ankle", (0.15, 1.0, 1.85))):
    row = []
    for d in ds:
        t0 = time.perf_counter()
        res = eqs_channel(f, ChannelConfig(path=path, distance=d))
        dt = time.perf_counter() - t0
        row.append(f"d={d:.2f}m {res['mag_db'][i_fc]:6.1f} dB")
    print(f"{path:15s} " + " | ".join(row) + f"   ({dt*1e3:.0f} ms / 240 freqs)")
print(f"closed-form capacitive divider: {eqs_channel_analytic(ChannelConfig()):.1f} dB")
for cret in (0.2e-12, 1e-12, 2e-12):
    c = ChannelConfig(c_ret_tx=cret, c_ret_rx=cret)
    print(f"  C_ret={cret*1e12:.1f} pF: MNA {eqs_channel(np.array([fc]), c)['mag_db'][0]:6.1f} dB, closed-form {eqs_channel_analytic(c):6.1f} dB")

res = eqs_channel(f, ChannelConfig())
vb = np.abs(res["body_potential"][i_fc])
print(f"body equipotential @500k: |V| spread across body nodes = {100*(vb.max()-vb.min())/vb.mean():.3f}%")
z = path_tissue_impedance(fc, "wrist_to_head", 0.95)[0]
print(f"tissue-only Z wrist->head @500k: {abs(z):.1f} Ohm  angle {np.degrees(np.angle(z)):.2f} deg")
r50 = eqs_channel(f, ChannelConfig(r_load=50.0))
print(f"capacitive, 50-Ohm termination @ 10k/500k/5M: "
      f"{r50['mag_db'][np.argmin(abs(f-1e4))]:.1f} / {r50['mag_db'][i_fc]:.1f} / {r50['mag_db'][np.argmin(abs(f-5e6))]:.1f} dB")
print(f"capacitive, high-Z termination @ 10k/500k/5M: "
      f"{res['mag_db'][np.argmin(abs(f-1e4))]:.1f} / {res['mag_db'][i_fc]:.1f} / {res['mag_db'][np.argmin(abs(f-5e6))]:.1f} dB")

# passband sanity: a 500 kHz tone must come out scaled by |H(fc)| with phase H(fc)
fs = 8e6
t = np.arange(16384) / fs
x = np.cos(2 * np.pi * fc * t)
yb = apply_channel(x, fs, channel_interpolator(res))
amp = 2 * abs(np.mean(yb[4000:-4000] * np.exp(-2j * np.pi * fc * t[4000:-4000])))
print(f"passband tone through channel: gain {20*np.log10(amp):.2f} dB (expect {res['mag_db'][i_fc]:.2f})")

# ---------------------------------------------------------------- 3. DCD-RLS ANC
print("\n== DCD-RLS adaptive noise cancellation ==")
rng = np.random.default_rng(7)
T, N = 20000, 16
# coloured reference (sEMG-like AR(2) process, eigenvalue spread ~1e3)
u = lfilter([1.0], [1.0, -1.6, 0.81], rng.standard_normal(T))
u /= u.std()
h = rng.standard_normal(N) * np.exp(-np.arange(N) / 4.0)   # unknown coupling path
v = lfilter(h, [1.0], u)
s = 0.1 * np.sign(np.sin(2 * np.pi * np.arange(T) / 50))     # data channel, buried ~15 dB under interference
d = s + v + 1e-3 * rng.standard_normal(T)
X = np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(N - 1), u], N)[:, ::-1]
print(f"reference eigenvalue spread: {np.ptp(np.log10(np.linalg.eigvalsh(X.T @ X))):.1f} decades")
print(f"input SIR: {10*np.log10(np.var(s)/np.var(v)):.1f} dB")

for Nu in (1, 2, 4, 8):
    flt = DCDRLS(n_taps=N, lam=0.999, delta=1e-2, H=1.0, Mb=16, Nu=Nu)
    flt.run(u[:100], d[:100])  # JIT warm-up
    t0 = time.perf_counter()
    out = flt.run(u, d, w_stride=10)
    dt = time.perf_counter() - t0
    mis = misalignment_db(out["W"], h)
    sir = 10 * np.log10(np.var(s[-5000:]) / np.var(out["e"][-5000:] - s[-5000:]))
    conv = np.argmax(mis < -20) * 10 if (mis < -20).any() else -1
    print(f"DCD-RLS Nu={Nu}: final misalign {mis[-1]:6.1f} dB, -20 dB reached @ n={conv:5d}, "
          f"output SIR {sir:5.1f} dB, {dt*1e6/T:.2f} us/sample")

e_r, w_r = rls(u, d, N, 0.999, 1e-2)
e_n, w_n = nlms(u, d, N, 0.5)
for name, e_, w_ in (("exact RLS", e_r, w_r), ("NLMS", e_n, w_n)):
    mis = 10 * np.log10(np.sum((w_ - h) ** 2) / np.sum(h ** 2))
    sir = 10 * np.log10(np.var(s[-5000:]) / np.var(e_[-5000:] - s[-5000:]))
    print(f"{name:9s}:    final misalign {mis:6.1f} dB, output SIR {sir:5.1f} dB")
