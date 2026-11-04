"""
BDN-04 :: Pipeline driver (Streamlit-free, cacheable, headless-testable)
========================================================================

Three entry points the dashboard calls, each a pure function of its args so it
can be wrapped in st.cache_data and exercised without a browser:

    run_storage(...)  -> DNA encoding + TdT yield readouts
    run_channel(...)  -> EQS-HBC Bode data
    run_link(...)     -> full wetware->edge link: modulate, bury, demod, ANC, RS

All heavy arrays returned for plotting are pre-decimated to <= ~3000 points so
Plotly stays at interactive framerates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from bdn04.tissue import (eqs_channel, ChannelConfig, eqs_channel_analytic,
                          path_tissue_impedance, path_length, PATHS)
from bdn04.dna_hfim import (DNACodec, DNAEncoded, TdTSynthesisSimulator,
                            HFIMModulator, HFIMConfig, MuscleConfig,
                            MuscleArtifactInjector, slice_symbols,
                            homopolymer_run_lengths)
from bdn04.dcd_rls import DCDRLS
from bdn04.ecc import ReedSolomonECC

_B = np.array(list("ACGT"))


def _decimate_plot(x: np.ndarray, max_pts: int = 3000) -> np.ndarray:
    if x.size <= max_pts:
        return x
    step = int(np.ceil(x.size / max_pts))
    return x[::step]


# =========================================================================== #
# Section 1 : DNA storage
# =========================================================================== #
def run_storage(text: str, N_synth: int, coupling_yield: float, p_base: float) -> dict:
    text = text or " "
    enc_naive = DNACodec.encode_naive(text)
    enc_rll = DNACodec.encode_rll(text)
    runs_naive = homopolymer_run_lengths(enc_naive.seq_int)
    runs_rll = homopolymer_run_lengths(enc_rll.seq_int)

    sim = TdTSynthesisSimulator(coupling_yield=coupling_yield, p_base=p_base)
    eta = sim.full_length_yield(N_synth)
    p_del_naive = sim.base_error_profile(enc_naive)

    return {
        "dna_naive": enc_naive.seq,
        "dna_rll": enc_rll.seq,
        "len_naive": len(enc_naive),
        "len_rll": len(enc_rll),
        "bits_per_char_naive": len(enc_naive) * 2.0 / max(len(text), 1),
        "bits_per_char_rll": len(enc_rll) * np.log2(3) / max(len(text), 1),
        "max_run_naive": int(runs_naive.max()) if runs_naive.size else 0,
        "max_run_rll": int(runs_rll.max()) if runs_rll.size else 0,
        "eta_N": float(eta),
        "mean_p_del_naive": float(p_del_naive.mean()) if p_del_naive.size else 0.0,
        "p_del_profile": _decimate_plot(p_del_naive),
        "run_profile": _decimate_plot(homopolymer_run_lengths(enc_naive.seq_int).astype(float)),
        "N_synth": N_synth,
        "coupling_yield": coupling_yield,
    }


# =========================================================================== #
# Section 2 : EQS-HBC channel Bode
# =========================================================================== #
def run_channel(path: str, distance: float, load_mode: str) -> dict:
    f = np.logspace(3, 7, 260)
    r_load = 50.0 if load_mode == "50ohm" else 10e6
    cfg = ChannelConfig(path=path, distance=distance, r_load=r_load)
    res = eqs_channel(f, cfg)
    i500 = int(np.argmin(np.abs(f - 500e3)))

    # tissue-only series impedance to the RX site (what distance actually moves)
    z_tissue = path_tissue_impedance(500e3, path, distance)[0]

    return {
        "f": f,
        "mag_db": res["mag_db"],
        "phase_deg": res["phase_deg"],
        "loss_500k_db": float(res["mag_db"][i500]),
        "analytic_db": float(eqs_channel_analytic(cfg)),
        "path_len": path_length(path),
        "distance": distance,
        "z_tissue_ohm": float(abs(z_tissue)),
        "z_tissue_phase": float(np.degrees(np.angle(z_tissue))),
        "load_mode": load_mode,
        "f_mark": 500e3,
        "mag_at_mark": float(res["mag_db"][i500]),
    }


# =========================================================================== #
# Section 3 + 4 : full link, ANC, Reed-Solomon recovery
# =========================================================================== #
@dataclass
class LinkParams:
    text: str = "A16Z_ACADEMY"
    nsym: int = 40
    mode: str = "motion"            # 'motion' | 'isometric'
    semg_depth: float = 3.0
    zmod_depth: float = 0.3
    snr_db: float = 25.0
    flex: bool = False              # "Flex Muscle" -> spike additive sEMG
    flex_gain: float = 2.2
    taps: int = 24
    lam: float = 0.995
    Nu: int = 8
    seed: int = 5


def run_link(p: LinkParams) -> dict:
    mod = HFIMModulator(HFIMConfig())
    ecc = ReedSolomonECC(nsym=p.nsym)
    coded, bits = ecc.encode_text(p.text)
    payload_bytes = len(p.text.encode("utf-8"))

    t, v_tx, b_nrz = mod.modulate(bits)
    semg_depth = p.semg_depth * (p.flex_gain if p.flex else 1.0)
    mcfg = MuscleConfig(mode=p.mode, semg_depth=semg_depth,
                        zmod_depth=p.zmod_depth, snr_db=p.snr_db)
    out = MuscleArtifactInjector(mcfg).apply_and_demodulate(
        t, v_tx, b_nrz, mod, np.random.default_rng(p.seed))

    d, r, s = out["primary_bb"], out["reference_bb"], out["data_bb_dec"]
    ob = out["osp_bb"]
    anc = DCDRLS(n_taps=p.taps, lam=p.lam, delta=1e-3, Nu=p.Nu)
    res = anc.run(r, d)
    e = res["e"]

    def sir(x):
        return 10 * np.log10(np.mean(s ** 2) / (np.mean((x - s) ** 2) + 1e-30))

    # convergence: settling time *after* the adaptation transient peaks (both
    # signals start near zero, so measuring from t=0 would false-trigger).
    err = np.abs(e - s)
    k = max(1, ob)
    env = np.convolve(err, np.ones(k) / k, mode="same")
    ss = np.median(env[-env.size // 4:])
    pk = int(np.argmax(env[:max(1, env.size // 2)]))        # transient peak (first half)
    tail = env[pk:] <= 1.5 * ss
    lock = pk + int(np.argmax(tail)) if tail.any() else -1

    raw_bits = slice_symbols(d, ob, bits.size)
    anc_bits = slice_symbols(e, ob, bits.size)
    raw_dec = ecc.decode_bits(raw_bits, coded)
    anc_dec = ecc.decode_bits(anc_bits, coded)

    # --- scope traces (decimated) ---
    osp = mod.cfg.osp
    rf_win = slice(0, min(6 * osp, v_tx.size))           # ~6 symbols of RF carrier
    tb = np.arange(d.size) / out["fs_bb"] * 1e3          # ms, baseband axis

    return {
        "coded_len": len(coded), "payload_bytes": payload_bytes,
        "code_rate": ecc.code_rate(payload_bytes), "t_capacity": ecc.t,
        "n_coded_bits": int(bits.size),
        "dna_tx": "".join(_B[(np.unpackbits(np.frombuffer(coded, np.uint8))
                              .reshape(-1, 2) * np.array([2, 1])).sum(1)]),
        "gamma_low": mod.gamma_low, "gamma_high": mod.gamma_high, "m": mod.m,
        # scope 1 : clean modulated RF
        "rf_t_us": _decimate_plot(t[rf_win] * 1e6),
        "rf_clean": _decimate_plot(v_tx[rf_win]),
        # scope 2 : baseband buried
        "bb_t_ms": _decimate_plot(tb),
        "bb_data": _decimate_plot(s),
        "bb_primary": _decimate_plot(d),
        # scope 3 : ANC recovery
        "bb_anc": _decimate_plot(e),
        "sir_in": sir(d), "sir_out": sir(e),
        "ber_raw": float((raw_bits != bits).mean()),
        "ber_anc": float((anc_bits != bits).mean()),
        "lock_samples": lock, "lock_ms": lock / out["fs_bb"] * 1e3 if lock >= 0 else -1,
        "n_updates_mean": float(res["n_updates"].mean()),
        "in_band_sir_db": out["in_band_sir_db"],
        "semg_depth_eff": semg_depth,
        # recovery
        "raw_ok": raw_dec.ok and raw_dec.text == p.text, "raw_text": raw_dec.text,
        "raw_byte_err": raw_dec.n_byte_errors,
        "anc_ok": anc_dec.ok and anc_dec.text == p.text, "anc_text": anc_dec.text,
        "anc_byte_err": anc_dec.n_byte_errors, "anc_corrected": anc_dec.corrected,
    }


if __name__ == "__main__":
    import json
    print("storage:", {k: v for k, v in run_storage("A16Z_ACADEMY", 500, 0.992, 0.004).items()
                        if not isinstance(v, np.ndarray)})
    for lm in ("high_z", "50ohm"):
        c = run_channel("wrist_to_head", 0.9, lm)
        print(f"channel {lm}: loss@500k {c['loss_500k_db']:.1f} dB (analytic {c['analytic_db']:.1f}), "
              f"tissue Z {c['z_tissue_ohm']:.0f} ohm")
    L = run_link(LinkParams())
    print(f"link: rate {L['code_rate']:.3f} t={L['t_capacity']} | SIR {L['sir_in']:+.1f}->{L['sir_out']:+.1f} dB"
          f" | BER {L['ber_raw']:.1%}->{L['ber_anc']:.1%} | lock {L['lock_ms']:.2f} ms"
          f" | RS-alone {'OK' if L['raw_ok'] else 'FAIL'} ({L['raw_byte_err']} be) |"
          f" ANC+RS {'OK' if L['anc_ok'] else 'FAIL'} '{L['anc_text']}' ({L['anc_byte_err']} be)")
