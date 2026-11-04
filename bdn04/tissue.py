"""
BDN-04 :: Transmission layer physics engine
===========================================

1. 4-pole Cole-Cole tissue dielectrics, Gabriel, Lau & Gabriel (1996),
   Phys. Med. Biol. 41:2271-2293 (Part III, Table 1):

       eps*(w) = eps_inf + sum_n  d_eps_n / (1 + (j w tau_n)^(1 - alpha_n))  +  sigma_i / (j w eps0)
       gamma(w) = j w eps0 eps*(w)          [complex admittivity, S/m]

2. EQS-HBC channel: the body path is a distributed RC ladder built from the
   Cole-Cole admittivity of concentric tissue layers (skin/fat/muscle/bone or
   brain). Each segment has a shunt capacitance to earth (body-to-ground).
   TX and RX are floating wearables whose grounds return through pF
   capacitances to earth. The whole circuit is solved by Modified Nodal
   Analysis (MNA), batched over frequency in one LAPACK call.

   Coupling: capacitive EQS-HBC (single signal electrode, floating ground
   plate). Galvanic coupling is deliberately NOT modelled: it depends on the 3D
   dipole fringe field across the limb cross-section, which a 1D ladder cannot
   represent, so any number it produced would be wrong.

   Valid for f <= ~10 MHz. At 10 MHz lambda_air = 30 m, and lambda in muscle
   is roughly 3 m, both well above body size, so the quasistatic assumption holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

EPS0 = 8.8541878128e-12
F_EQS_MAX = 10e6


# --------------------------------------------------------------------------- #
# 1. Cole-Cole tissue models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColeColeTissue:
    name: str
    eps_inf: float
    d_eps: tuple[float, float, float, float]
    tau: tuple[float, float, float, float]      # seconds
    alpha: tuple[float, float, float, float]
    sigma_i: float                               # static ionic conductivity, S/m

    def eps_complex(self, f) -> np.ndarray:
        """Complex relative permittivity eps' - j eps'' (ionic term included)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        w = 2.0 * np.pi * f
        de = np.asarray(self.d_eps)
        tau = np.asarray(self.tau)
        a = np.asarray(self.alpha)
        jwt = 1j * w[:, None] * tau[None, :]
        disp = np.sum(de / (1.0 + jwt ** (1.0 - a)), axis=1)
        return self.eps_inf + disp + self.sigma_i / (1j * w * EPS0)

    def admittivity(self, f) -> np.ndarray:
        """gamma = sigma + j w eps0 eps_r   [S/m]."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        return 1j * 2.0 * np.pi * f * EPS0 * self.eps_complex(f)

    def sigma(self, f) -> np.ndarray:
        return self.admittivity(f).real

    def eps_r(self, f) -> np.ndarray:
        return self.eps_complex(f).real


ps, ns, us, ms = 1e-12, 1e-9, 1e-6, 1e-3

# Gabriel 1996, Part III, Table 1 (verbatim parameters)
TISSUES: dict[str, ColeColeTissue] = {
    "muscle": ColeColeTissue("Muscle", 4.0, (50.0, 7000.0, 1.2e6, 2.5e7),
                             (7.234 * ps, 353.678 * ns, 318.310 * us, 2.274 * ms),
                             (0.10, 0.10, 0.10, 0.00), 0.200),
    "fat": ColeColeTissue("Fat (not infiltrated)", 2.5, (3.0, 15.0, 3.3e4, 1.0e7),
                          (7.958 * ps, 15.915 * ns, 159.155 * us, 7.958 * ms),
                          (0.20, 0.10, 0.05, 0.01), 0.010),
    "skin_dry": ColeColeTissue("Skin (dry)", 4.0, (32.0, 1100.0, 0.0, 0.0),
                               (7.234 * ps, 32.481 * ns, 159.155 * us, 15.915 * ms),
                               (0.00, 0.20, 0.20, 0.20), 0.0002),
    "skin_wet": ColeColeTissue("Skin (wet)", 4.0, (39.0, 280.0, 3.0e4, 3.0e4),
                               (7.958 * ps, 79.577 * ns, 1.592 * us, 1.592 * ms),
                               (0.10, 0.00, 0.16, 0.20), 0.0004),
    "bone": ColeColeTissue("Bone (cortical)", 2.5, (10.0, 180.0, 5000.0, 1.0e5),
                           (13.263 * ps, 79.577 * ns, 159.155 * us, 15.915 * ms),
                           (0.20, 0.20, 0.20, 0.00), 0.020),
    "blood": ColeColeTissue("Blood", 4.0, (56.0, 5200.0, 0.0, 0.0),
                            (8.377 * ps, 132.629 * ns, 159.155 * us, 15.915 * ms),
                            (0.10, 0.10, 0.20, 0.00), 0.700),
    "grey_matter": ColeColeTissue("Brain (grey matter)", 4.0, (45.0, 400.0, 2.0e5, 4.5e7),
                                  (7.958 * ps, 15.915 * ns, 106.103 * us, 5.305 * ms),
                                  (0.10, 0.15, 0.22, 0.00), 0.020),
}


# --------------------------------------------------------------------------- #
# 2. Body geometry: concentric-layer cylinders along a transmission path
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BodyRegion:
    """Concentric cylinder. layers = outside -> inside: (tissue_key, thickness_m);
    the final entry is the core and its thickness is ignored (fills to r = 0)."""
    name: str
    radius: float
    layers: tuple[tuple[str, float], ...]

    def layer_areas(self) -> list[tuple[str, float]]:
        out, r_out = [], self.radius
        for i, (tissue, t) in enumerate(self.layers):
            r_in = 0.0 if i == len(self.layers) - 1 else max(r_out - t, 0.0)
            out.append((tissue, np.pi * (r_out ** 2 - r_in ** 2)))
            r_out = r_in
        return out

    def admittance_per_length(self, f) -> np.ndarray:
        """Y' = sum_i gamma_i * A_i   [S*m]. Longitudinal current; layers in parallel."""
        return sum(TISSUES[t].admittivity(f) * A for t, A in self.layer_areas())


SKIN = 1.5e-3
REGIONS: dict[str, BodyRegion] = {
    "forearm":   BodyRegion("Forearm",   0.035, (("skin_wet", SKIN), ("fat", 4e-3),  ("muscle", 0.019), ("bone", 0))),
    "upper_arm": BodyRegion("Upper arm", 0.045, (("skin_wet", SKIN), ("fat", 6e-3),  ("muscle", 0.0255), ("bone", 0))),
    "torso":     BodyRegion("Torso",     0.150, (("skin_wet", SKIN), ("fat", 15e-3), ("muscle", 0.1035), ("bone", 0))),
    "neck":      BodyRegion("Neck",      0.055, (("skin_wet", SKIN), ("fat", 5e-3),  ("muscle", 0.0335), ("bone", 0))),
    "head":      BodyRegion("Head",      0.090, (("skin_wet", SKIN), ("fat", 3e-3),  ("bone", 6e-3),    ("grey_matter", 0))),
    "thigh":     BodyRegion("Thigh",     0.080, (("skin_wet", SKIN), ("fat", 12e-3), ("muscle", 0.0525), ("bone", 0))),
    "shank":     BodyRegion("Shank",     0.050, (("skin_wet", SKIN), ("fat", 6e-3),  ("muscle", 0.0275), ("bone", 0))),
}

# (region, segment length in m), ordered from the TX (left wrist) outward
PATHS: dict[str, tuple[tuple[str, float], ...]] = {
    "wrist_to_head":  (("forearm", 0.26), ("upper_arm", 0.30), ("torso", 0.20), ("neck", 0.12), ("head", 0.12)),
    "wrist_to_wrist": (("forearm", 0.26), ("upper_arm", 0.30), ("torso", 0.36), ("upper_arm", 0.30), ("forearm", 0.26)),
    "wrist_to_ankle": (("forearm", 0.26), ("upper_arm", 0.30), ("torso", 0.50), ("thigh", 0.45), ("shank", 0.42)),
}


def path_length(path: str) -> float:
    return float(sum(L for _, L in PATHS[path]))


def _discretize(path: str, n_seg: int):
    """Uniform segmentation; returns per-segment region keys, dl, and node positions."""
    L = path_length(path)
    dl = L / n_seg
    mids = (np.arange(n_seg) + 0.5) * dl
    bounds = np.cumsum([Lr for _, Lr in PATHS[path]])
    idx = np.searchsorted(bounds, mids, side="right").clip(max=len(bounds) - 1)
    regions = [PATHS[path][i][0] for i in idx]
    return regions, dl, np.arange(n_seg + 1) * dl


def path_tissue_impedance(f, path: str, d: float, n_seg: int = 200) -> np.ndarray:
    """Pure volume-conductor series impedance from TX site to distance d [Ohm],
    with no return path. This is the quantity shown on the 'tissue Bode' plot."""
    f = np.atleast_1d(np.asarray(f, float))
    regions, dl, x = _discretize(path, n_seg)
    k = int(np.clip(np.round(d / dl), 1, n_seg))
    Yp = {r: REGIONS[r].admittance_per_length(f) for r in set(regions[:k])}
    return sum(dl / Yp[r] for r in regions[:k])


# --------------------------------------------------------------------------- #
# 3. Electrode-skin interface: transverse stratum corneum + double layer (CPE)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Electrode:
    area: float = 1e-4            # m^2 (1 cm^2)
    r_ct: float = 2.0e4           # charge-transfer resistance, Ohm
    c_dl: float = 5.0e-8          # double-layer capacitance, F
    cpe_beta: float = 0.8         # constant-phase exponent
    r_series: float = 10.0        # lead/gel, Ohm
    sc_thickness: float = 20e-6   # stratum corneum thickness, m (transverse crossing)

    def impedance(self, f) -> np.ndarray:
        f = np.atleast_1d(np.asarray(f, float))
        w = 2 * np.pi * f
        z_dl = self.r_ct / (1.0 + (1j * w * self.r_ct * self.c_dl) ** self.cpe_beta)
        z_sc = self.sc_thickness / (TISSUES["skin_dry"].admittivity(f) * self.area)
        return self.r_series + z_dl + z_sc


ELECTRODES = {
    "wet_AgAgCl": Electrode(),
    "dry_metal": Electrode(r_ct=1.0e6, c_dl=1.0e-9, cpe_beta=0.85, r_series=20.0, sc_thickness=40e-6),
}


# --------------------------------------------------------------------------- #
# 4. EQS-HBC channel via batched Modified Nodal Analysis
# --------------------------------------------------------------------------- #
@dataclass
class ChannelConfig:
    path: str = "wrist_to_head"
    distance: float = 0.9               # m along path from TX
    electrode: str = "wet_AgAgCl"
    n_seg: int = 48
    c_body_total: float = 150e-12       # whole-body capacitance to earth
    body_surface: float = 1.8           # m^2, used to apportion C_body along the path
    c_ret_tx: float = 0.5e-12           # TX ground plate to earth
    c_ret_rx: float = 0.5e-12           # RX ground plate to earth
    c_gnd_body: float = 0.0             # parasitic ground-plate to body (capacitive mode)
    r_source: float = 50.0
    r_load: float = 10e6                # 50 Ohm here shows the classic high-pass trap
    c_load: float = 5e-12


def _stamp(Y, a, b, y):
    """Stamp admittance y[F] between nodes a, b (None = earth) into Y[F, n, n]."""
    if a is not None:
        Y[:, a, a] += y
    if b is not None:
        Y[:, b, b] += y
    if a is not None and b is not None:
        Y[:, a, b] -= y
        Y[:, b, a] -= y


def eqs_channel(f, cfg: ChannelConfig) -> dict:
    """Solve the full TX-body-RX circuit. Returns H = V_rx / V_tx and diagnostics.

    Closed-form sanity limit (high-Z RX, C_body >> C_ret, tissue Z << return reactance):
        |H| ~= (C_ret_tx / C_body) * C_ret_rx / (C_ret_rx + C_load)
    """
    f = np.atleast_1d(np.asarray(f, float))
    if f.max() > 3 * F_EQS_MAX:
        raise ValueError("Above ~30 MHz the quasistatic model is invalid (body resonance/antenna regime).")
    w = 2 * np.pi * f
    F = f.size
    regions, dl, xs = _discretize(cfg.path, cfg.n_seg)
    nb = cfg.n_seg + 1

    # node map
    TX_OUT, TX_GND, TX_PAD, RX_PAD, RX_GND = range(5)
    B0 = 5
    n_nodes = B0 + nb
    I_VS = n_nodes                       # extra MNA row for the voltage source current
    N = n_nodes + 1
    Y = np.zeros((F, N, N), dtype=complex)

    k_rx = int(np.clip(np.round(cfg.distance / dl), 1, cfg.n_seg))

    # tissue ladder
    Yp = {r: REGIONS[r].admittance_per_length(f) for r in set(regions)}
    for i, r in enumerate(regions):
        _stamp(Y, B0 + i, B0 + i + 1, Yp[r] / dl)

    # body-to-earth capacitance, apportioned by surface area; the residual
    # (other limbs, legs, etc.) lumps at the first torso node
    surf = np.array([2 * np.pi * REGIONS[r].radius * dl for r in regions])
    node_surf = np.zeros(nb)
    node_surf[:-1] += surf / 2
    node_surf[1:] += surf / 2
    c_node = cfg.c_body_total * node_surf / cfg.body_surface
    resid = cfg.c_body_total - c_node.sum()
    if resid > 0:
        torso = [i for i, r in enumerate(regions) if r == "torso"]
        c_node[torso[0] if torso else nb // 2] += resid
    for i in range(nb):
        _stamp(Y, B0 + i, None, 1j * w * c_node[i])

    ze = ELECTRODES[cfg.electrode].impedance(f)

    # TX
    _stamp(Y, TX_OUT, TX_PAD, np.full(F, 1 / cfg.r_source, complex))
    _stamp(Y, TX_PAD, B0, 1 / ze)
    _stamp(Y, TX_GND, None, 1j * w * cfg.c_ret_tx)
    # RX
    _stamp(Y, B0 + k_rx, RX_PAD, 1 / ze)
    _stamp(Y, RX_PAD, RX_GND, 1 / cfg.r_load + 1j * w * cfg.c_load)
    _stamp(Y, RX_GND, None, 1j * w * cfg.c_ret_rx)

    if cfg.c_gnd_body > 0:
        _stamp(Y, TX_GND, B0, 1j * w * cfg.c_gnd_body)
        _stamp(Y, RX_GND, B0 + k_rx, 1j * w * cfg.c_gnd_body)

    # tiny leakage keeps floating nodes well-conditioned at low f
    for n in range(n_nodes):
        Y[:, n, n] += 1e-15

    # ideal source: V(TX_OUT) - V(TX_GND) = 1
    Y[:, TX_OUT, I_VS] += 1
    Y[:, TX_GND, I_VS] -= 1
    Y[:, I_VS, TX_OUT] += 1
    Y[:, I_VS, TX_GND] -= 1
    rhs = np.zeros((F, N, 1), dtype=complex)
    rhs[:, I_VS, 0] = 1.0

    v = np.linalg.solve(Y, rhs)[..., 0]
    H = v[:, RX_PAD] - v[:, RX_GND]
    v_body = v[:, B0:B0 + nb] - v[:, [TX_GND]]   # body potential seen from the TX reference

    return {
        "f": f,
        "H": H,
        "mag_db": 20 * np.log10(np.abs(H) + 1e-300),
        "phase_deg": np.degrees(np.unwrap(np.angle(H))),
        "group_delay_s": (-np.gradient(np.unwrap(np.angle(H)), w) if F > 1 else np.full(1, np.nan)),
        "body_potential": v_body,        # [F, nodes]: equipotential check
        "node_x": xs,
        "rx_node": k_rx,
        "c_node": c_node,
    }


def channel_interpolator(res: dict):
    """Interpolate H(f) in log-f (|H| in dB, unwrapped phase) for arbitrary FFT bins."""
    lf = np.log10(res["f"])
    mag = res["mag_db"]
    ph = np.unwrap(np.angle(res["H"]))

    def H_of(fq):
        fq = np.asarray(fq, float)
        out = np.zeros(fq.shape, complex)
        nz = fq > 0
        lq = np.log10(fq[nz])
        m = np.interp(lq, lf, mag)
        p = np.interp(lq, lf, ph)
        out[nz] = 10 ** (m / 20) * np.exp(1j * p)
        return out

    return H_of


def apply_channel(x: np.ndarray, fs: float, H_of) -> np.ndarray:
    """Pass a real passband signal through H via a zero-padded FFT (linear convolution)."""
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    fk = np.fft.rfftfreq(nfft, 1 / fs)
    return np.fft.irfft(X * H_of(fk), nfft)[:n]


def eqs_channel_analytic(cfg: ChannelConfig) -> float:
    """Flat-band capacitive-divider limit, in dB."""
    g = (cfg.c_ret_tx / (cfg.c_body_total + cfg.c_ret_tx)) * cfg.c_ret_rx / (cfg.c_ret_rx + cfg.c_load)
    return 20 * np.log10(g)
