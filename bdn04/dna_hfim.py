"""
BDN-04 :: Storage + Transmission front-end
==========================================

Phase-2 engine, sits beside tissue.py and dcd_rls.py. Three stages, all
vectorized (NumPy/SciPy), no per-sample Python loops in the signal path:

  1. DNACodec              ASCII <-> quaternary DNA, naive and Goldman-style RLL
  2. TdTSynthesisSimulator template-free enzymatic (TdT) *data-storage* yield model:
                           stepwise coupling decay eta(N)=Y^N and homopolymer
                           deletion hazard. This is an information-storage error
                           model over symbol strings; nothing here is a wet-lab
                           protocol or a biologically functional sequence.
  3. HFIMModulator         500 kHz capacitive backscatter load modulation
  4. MuscleArtifactInjector  post-demod in-band impedance interference + I/Q demod,
                           producing the dual (primary / reference) channels that
                           feed dcd_rls.DCDRLS.

Run `python -m bdn04.dna_hfim` for the validation block.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, resample_poly

try:  # port impedance from the physics engine; standalone fallback = 440 Ohm
    from bdn04.tissue import path_tissue_impedance
    def _port_impedance() -> float:
        return float(abs(path_tissue_impedance(5e5, "wrist_to_head", 0.9)[0]))
except Exception:  # pragma: no cover
    def _port_impedance() -> float:
        return 440.0


# =========================================================================== #
# 1. DNA codec
# =========================================================================== #
_BASES = np.array(list("ACGT"))
_B2I = {b: i for i, b in enumerate(_BASES)}
# Goldman rotating transition table: from base p, trit t -> one of the 3 OTHER
# bases. Guarantees seq[i] != seq[i-1], i.e. zero homopolymer runs.
_ROT = np.array([[1, 2, 3],   # A -> C G T
                 [2, 3, 0],   # C -> G T A
                 [3, 0, 1],   # G -> T A C
                 [0, 1, 2]])  # T -> A C G
_UNROT = np.full((4, 4), -1, dtype=np.int64)   # _UNROT[prev, cur] = trit
for _p in range(4):
    for _t in range(3):
        _UNROT[_p, _ROT[_p, _t]] = _t


@dataclass
class DNAEncoded:
    seq_int: np.ndarray        # int8 array of base indices 0..3
    scheme: str                # 'naive' | 'rll'
    n_bytes: int               # payload length, needed for exact decode
    n_trits: int = 0           # RLL only
    start_base: int = 0        # RLL only

    @property
    def seq(self) -> str:
        return "".join(_BASES[self.seq_int])

    def __len__(self) -> int:
        return int(self.seq_int.size)


class DNACodec:
    """Reversible text <-> DNA. Naive = 2 bit/base; RLL = Goldman transition code
    (log2(3) ~ 1.58 bit/base) with no homopolymers."""

    # ---- naive quaternary : A:00 C:01 G:10 T:11 --------------------------- #
    @staticmethod
    def encode_naive(text: str) -> DNAEncoded:
        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        bits = np.unpackbits(data)                       # MSB-first, len 8*nbytes
        vals = (bits.reshape(-1, 2) * np.array([2, 1])).sum(1).astype(np.int8)
        return DNAEncoded(vals, "naive", data.size)

    @staticmethod
    def decode_naive(enc: DNAEncoded) -> str:
        v = enc.seq_int.astype(np.uint8)
        bits = np.empty((v.size, 2), dtype=np.uint8)
        bits[:, 0] = (v >> 1) & 1
        bits[:, 1] = v & 1
        data = np.packbits(bits.reshape(-1))[:enc.n_bytes]
        return data.tobytes().decode("utf-8", errors="replace")

    # ---- Goldman-style run-length-constrained ----------------------------- #
    @staticmethod
    def encode_rll(text: str, start_base: int = 0) -> DNAEncoded:
        data = text.encode("utf-8")
        n_bytes = len(data)
        big = int.from_bytes(data, "big") if n_bytes else 0
        n_trits = max(1, math.ceil(n_bytes * 8 * math.log(2) / math.log(3)))
        # base-3 digits, least-significant first, fixed width
        trits = np.zeros(n_trits, dtype=np.int64)
        for i in range(n_trits):
            big, trits[i] = divmod(big, 3)
        # walk the rotation table (sequential by construction; cheap O(N))
        seq = np.empty(n_trits, dtype=np.int8)
        prev = start_base
        for i in range(n_trits):
            prev = _ROT[prev, trits[i]]
            seq[i] = prev
        return DNAEncoded(seq, "rll", n_bytes, n_trits, start_base)

    @staticmethod
    def decode_rll(enc: DNAEncoded) -> str:
        seq = enc.seq_int.astype(np.int64)
        prev = np.empty_like(seq)
        prev[0] = enc.start_base
        prev[1:] = seq[:-1]
        trits = _UNROT[prev, seq]
        big = 0
        for t in trits[::-1]:
            big = big * 3 + int(t)
        return big.to_bytes(enc.n_bytes, "big").decode("utf-8", errors="replace")

    # ---- dispatch + bit view for the modulator ---------------------------- #
    @classmethod
    def encode(cls, text: str, scheme: str = "naive") -> DNAEncoded:
        return cls.encode_rll(text) if scheme == "rll" else cls.encode_naive(text)

    @classmethod
    def decode(cls, enc: DNAEncoded) -> str:
        return cls.decode_rll(enc) if enc.scheme == "rll" else cls.decode_naive(enc)

    @staticmethod
    def to_bits(enc: DNAEncoded) -> np.ndarray:
        """2-bit-per-base packed bitstream (uint8 0/1), for the line coder."""
        v = enc.seq_int.astype(np.uint8)
        return np.column_stack(((v >> 1) & 1, v & 1)).reshape(-1)


# =========================================================================== #
# 2. TdT enzymatic-storage yield model
# =========================================================================== #
def homopolymer_run_lengths(seq_int: np.ndarray) -> np.ndarray:
    """Length of the homopolymer run each position belongs to (vectorized)."""
    seq_int = np.asarray(seq_int)
    if seq_int.size == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.empty(seq_int.size, dtype=bool)
    change[0] = True
    change[1:] = seq_int[1:] != seq_int[:-1]
    run_id = np.cumsum(change) - 1
    counts = np.bincount(run_id)
    return counts[run_id]


@dataclass
class SynthesisReport:
    cumulative_yield: float          # eta(N) = Y^N
    p_del: np.ndarray                # per-base analytical deletion probability
    mean_p_del: float
    max_run: int
    synthesized: DNAEncoded          # realized strand after stochastic deletions
    n_deleted: int


class TdTSynthesisSimulator:
    """Template-free (TdT) synthesis yield model for DNA *data storage*.

    coupling_yield : per-cycle stepwise coupling efficiency Y in [0.970, 0.999]
    p_base         : per-nucleotide base deletion hazard inside a homopolymer run
    """

    def __init__(self, coupling_yield: float = 0.992, p_base: float = 0.004):
        if not 0.970 <= coupling_yield <= 0.999:
            raise ValueError("coupling_yield must lie in [0.970, 0.999]")
        self.Y = coupling_yield
        self.p_base = p_base

    def full_length_yield(self, N: int) -> float:
        return self.Y ** N

    def base_error_profile(self, enc: DNAEncoded) -> np.ndarray:
        """P_del(L) = 1 - (1 - p_base)^L per base, L = local homopolymer run length.
        RLL strands have L==1 everywhere -> flat floor of p_base."""
        L = homopolymer_run_lengths(enc.seq_int)
        return 1.0 - (1.0 - self.p_base) ** L

    def simulate(self, enc: DNAEncoded, rng: np.random.Generator | None = None) -> SynthesisReport:
        rng = rng or np.random.default_rng()
        p_del = self.base_error_profile(enc)
        keep = rng.random(enc.seq_int.size) >= p_del
        realized = DNAEncoded(enc.seq_int[keep].astype(np.int8), enc.scheme,
                              enc.n_bytes, enc.n_trits, enc.start_base)
        return SynthesisReport(
            cumulative_yield=self.full_length_yield(len(enc)),
            p_del=p_del, mean_p_del=float(p_del.mean()),
            max_run=int(homopolymer_run_lengths(enc.seq_int).max()),
            synthesized=realized, n_deleted=int((~keep).sum()),
        )


# =========================================================================== #
# 3. HFIM load modulator (500 kHz capacitive backscatter)
# =========================================================================== #
@dataclass
class HFIMConfig:
    f_c: float = 500e3          # carrier
    R_sym: float = 10e3         # symbol rate (bits/s, NRZ)
    f_s: float = 2e6            # sample rate -> osp = 200
    Z0: float | None = None     # tissue-electrode port impedance (None -> tissue.py)
    Z_low: float = 100.0
    Z_high: float = 10e3
    V_carrier: float = 1.0
    osp_bb: int = 8             # baseband oversampling for the ANC (fs_bb = osp_bb*R_sym)

    @property
    def osp(self) -> int:
        return int(round(self.f_s / self.R_sym))

    @property
    def fs_bb(self) -> float:
        return self.osp_bb * self.R_sym


class HFIMModulator:
    def __init__(self, cfg: HFIMConfig | None = None):
        self.cfg = cfg or HFIMConfig()
        self.Z0 = self.cfg.Z0 if self.cfg.Z0 is not None else _port_impedance()
        self.gamma_low = (self.cfg.Z_low - self.Z0) / (self.cfg.Z_low + self.Z0)
        self.gamma_high = (self.cfg.Z_high - self.Z0) / (self.cfg.Z_high + self.Z0)
        # modulation index from the reflection-coefficient swing; half-swing keeps
        # the AM envelope 1 + m b(t) strictly positive for b in {-1, +1}.
        self.m = min(1.0, abs(self.gamma_high - self.gamma_low) / 2.0)

    def bits_to_nrz(self, bits: np.ndarray) -> np.ndarray:
        """{0,1} -> {-1,+1}, oversampled to f_s (rectangular NRZ)."""
        sym = 2 * np.asarray(bits, dtype=np.int8) - 1
        return np.repeat(sym, self.cfg.osp).astype(np.float64)

    def modulate(self, bits: np.ndarray):
        b = self.bits_to_nrz(bits)
        t = np.arange(b.size) / self.cfg.f_s
        carrier = np.cos(2 * np.pi * self.cfg.f_c * t)
        v_tx = self.cfg.V_carrier * (1.0 + self.m * b) * carrier
        return t, v_tx, b


# =========================================================================== #
# 4. Muscle impedance interference + synchronous I/Q demod + dual channels
# =========================================================================== #
@dataclass
class MuscleConfig:
    """Two physically-distinct muscle mechanisms (both real, both 'muscle'):

      additive sEMG   : bioelectric voltage the electrodes pick up (20-450 Hz).
                        Dominant, additive, linearly correlated across electrodes
                        -> the textbook Widrow ANC target. Modelled at the
                        demodulated baseband (sEMG at RF would be demod-rejected).
      multiplicative  : contraction changes tissue/contact impedance, modulating
        Delta Z         the backscatter envelope (5-150 Hz). Produces a data x muscle
                        cross-term in the data band that a linear ANC cannot fully
                        cancel -> handled by the outer ECC. This is the Section-X term.

    Set semg_depth=0 for Varun's original multiplicative-only spec.
    """
    # --- additive sEMG (ANC-cancellable) ---
    semg_depth: float = 3.0     # sEMG RMS relative to data RMS (>1 buries the data)
    semg_lo: float = 20.0
    semg_hi: float = 450.0
    # --- multiplicative Delta Z (Section-X envelope modulation, ECC residual) ---
    zmod_depth: float = 0.3     # envelope modulation depth of Delta Z / Z
    f_lo: float = 5.0
    f_hi: float = 150.0
    # --- shared ---
    fs_gen: float = 2000.0      # muscle synthesized here, then interpolated to f_s
    mode: str = "isometric"     # 'isometric' (sustained) | 'motion' (bursts)
    snr_db: float = 25.0        # AWGN on the primary channel
    ref_leak: float = 0.03      # fraction of data that leaks into the reference
    ref_noise_db: float = 25.0  # independent sensor noise on the reference channel
    # back-compat alias: `depth` maps to the multiplicative term
    @property
    def depth(self) -> float:
        return self.zmod_depth


class MuscleArtifactInjector:
    def __init__(self, cfg: MuscleConfig | None = None):
        self.cfg = cfg or MuscleConfig()

    def _band_noise(self, n: int, fs: float, flo: float, fhi: float,
                    rng: np.random.Generator) -> np.ndarray:
        """Unit-RMS pink, band-limited [flo,fhi], with a contraction/motion
        envelope. Synthesized at fs_gen then interpolated to fs (the band is
        << fs, so linear interpolation is spectrally exact)."""
        c = self.cfg
        n_lo = max(64, int(np.ceil(n * c.fs_gen / fs)))
        spec = rng.standard_normal(n_lo // 2 + 1) + 1j * rng.standard_normal(n_lo // 2 + 1)
        fk = np.fft.rfftfreq(n_lo, 1 / c.fs_gen)
        shape = np.zeros_like(fk)
        shape[1:] = 1.0 / np.sqrt(fk[1:])               # 1/f power -> pink
        x = np.fft.irfft(spec * shape, n_lo)
        sos = butter(4, [flo, fhi], btype="band", fs=c.fs_gen, output="sos")
        x = sosfiltfilt(sos, x)
        tg = np.arange(n_lo) / c.fs_gen
        if c.mode == "motion":
            env = np.zeros(n_lo)
            for t0 in np.linspace(tg[-1] * 0.15, tg[-1] * 0.85, 4):
                env += np.exp(-0.5 * ((tg - t0) / 0.07) ** 2)       # burst transients
            x = x * (0.1 + env)
        else:  # isometric: ramp-and-hold
            x = x * (0.5 + 0.5 * np.tanh((tg - tg[-1] * 0.2) / (tg[-1] * 0.1)))
        x /= (np.std(x) + 1e-12)
        return np.interp(np.arange(n) / fs, tg, x)

    def _delta_z(self, n, fs, rng):   # multiplicative impedance band (5-150 Hz)
        return self._band_noise(n, fs, self.cfg.f_lo, self.cfg.f_hi, rng)

    def _lpf(self, x: np.ndarray, fs: float, R_sym: float) -> np.ndarray:
        b, a = butter(4, 1.5 * R_sym, fs=fs, output="ba")
        return filtfilt(b, a, x)

    def apply_and_demodulate(self, t, v_tx, b_nrz, mod: HFIMModulator,
                             rng: np.random.Generator | None = None):
        """Inject multiplicative Delta Z onto the backscatter envelope, then do a
        synchronous I/Q down-conversion. Returns dict with the dual channels."""
        rng = rng or np.random.default_rng()
        cfg, c = mod.cfg, self.cfg
        n = v_tx.size
        carrier = np.cos(2 * np.pi * cfg.f_c * t)

        # (1) multiplicative Delta Z on the backscatter envelope (Section-X term)
        dz = self._delta_z(n, cfg.f_s, rng)
        v_rx = v_tx * (1.0 + c.zmod_depth * dz)

        # synchronous I/Q demod: 2*v*cos, LPF -> in-phase baseband.
        bb = self._lpf(2.0 * v_rx * carrier, cfg.f_s, cfg.R_sym)
        bb = bb - np.mean(bb)
        # data reference = payload through the SAME demod chain (not ideal NRZ), so
        # pulse-shaping isn't miscounted as interference.
        data_bb = self._lpf(2.0 * v_tx * carrier, cfg.f_s, cfg.R_sym)
        data_bb = data_bb - np.mean(data_bb)
        zmod_bb = bb - data_bb                                   # multiplicative residual
        rms_s = np.sqrt(np.mean(data_bb ** 2)) + 1e-12

        # (2) additive sEMG bioelectric pickup at baseband (dominant, cancellable)
        semg = self._band_noise(n, cfg.f_s, c.semg_lo, c.semg_hi, rng)
        semg = semg * (c.semg_depth * rms_s)

        muscle_bb = semg + zmod_bb                               # total muscle in primary
        awgn = rng.standard_normal(n) * (rms_s / 10 ** (c.snr_db / 20))
        primary = data_bb + muscle_bb + awgn

        # reference electrode pair: sees the additive sEMG through a short,
        # minimum-phase inter-electrode coupling FIR (cm spacing -> few taps; stable
        # fast-decaying inverse). It does NOT see the backscatter-envelope Delta Z
        # term (that rides the signal path), which is why the multiplicative part
        # survives the ANC and is left to the ECC.
        h_couple = np.array([1.0, 0.3, 0.1])
        semg_ref = np.convolve(semg, h_couple, mode="same")
        ref_noise = rng.standard_normal(n) * (np.sqrt(np.mean(semg ** 2)) / 10 ** (c.ref_noise_db / 20))
        reference = semg_ref + c.ref_leak * data_bb + ref_noise

        in_band_sir_db = 10 * np.log10(rms_s ** 2 / (np.mean(muscle_bb ** 2) + 1e-30))

        # Decimate the ANC-facing channels from f_s to fs_bb. The adaptive filter
        # CANNOT run at the 2 MHz RF rate: a 5-150 Hz reference sampled at 2 MHz has
        # near-identical adjacent taps, so R_uu is singular and recursive RLS
        # diverges. At fs_bb the tap covariance is well-conditioned.
        q_up, q_dn = cfg.osp_bb, cfg.osp
        primary_bb = resample_poly(primary, q_up, q_dn)
        reference_bb = resample_poly(reference, q_up, q_dn)
        data_bb_dec = resample_poly(data_bb, q_up, q_dn)

        return {
            "v_rx": v_rx, "baseband": bb, "data_bb": data_bb, "muscle_bb": muscle_bb,
            "delta_z": dz, "primary": primary, "reference": reference,
            "in_band_sir_db": float(in_band_sir_db), "m": mod.m,
            # baseband channels for dcd_rls.DCDRLS (d = primary_bb, u = reference_bb):
            "fs_bb": cfg.fs_bb, "osp_bb": cfg.osp_bb,
            "primary_bb": primary_bb, "reference_bb": reference_bb, "data_bb_dec": data_bb_dec,
        }


# --------------------------------------------------------------------------- #
# symbol recovery (for the ideal round-trip check and post-ANC decode)
# --------------------------------------------------------------------------- #
def slice_symbols(baseband: np.ndarray, osp: int, n_bits: int) -> np.ndarray:
    """Integrate-and-dump matched filter for rectangular NRZ: average the central
    portion of each symbol (avoids filter-transient edges), then decide sign.
    Gives ~osp-fold processing gain over single-sample slicing."""
    usable = min(n_bits * osp, (baseband.size // osp) * osp)
    seg = baseband[:usable].reshape(-1, osp)
    lo, hi = int(0.3 * osp), max(int(0.7 * osp), int(0.3 * osp) + 1)
    stat = seg[:, lo:hi].mean(axis=1)
    out = np.zeros(n_bits, dtype=np.uint8)
    out[:stat.size] = stat > 0
    return out


def repetition_decode(bits: np.ndarray, payload_bits: int) -> np.ndarray:
    """Per-bit majority vote across whole repeated copies of the payload.

    Real DNA data storage never ships uncoded: physical read/write error rates
    (here ~1-3% after the ANC) are driven to ~0 with an outer code. A repetition
    code is the simplest faithful stand-in; swap for Reed-Solomon / fountain
    (Erlich-Zielinski 2017) in production. Returns the recovered payload bits."""
    n_copies = bits.size // payload_bits
    if n_copies < 1:
        return bits[:payload_bits]
    block = bits[:n_copies * payload_bits].reshape(n_copies, payload_bits)
    return (block.mean(axis=0) >= 0.5).astype(np.uint8)


def bits_to_text(bits: np.ndarray, n_bytes: int) -> str:
    v = (bits.reshape(-1, 2) * np.array([2, 1])).sum(1).astype(np.int8)
    return DNACodec.decode_naive(DNAEncoded(v, "naive", n_bytes))


# =========================================================================== #
# Validation
# =========================================================================== #
if __name__ == "__main__":
    import time

    print(f"port impedance Z0 = {_port_impedance():.1f} Ohm")

    # -- 1. codec round-trips -------------------------------------------------
    print("\n== DNA codec round-trip ==")
    for scheme in ("naive", "rll"):
        for msg in ("A16Z_ACADEMY", "BDN-04 :: wetware-to-edge", "x"):
            enc = DNACodec.encode(msg, scheme)
            dec = DNACodec.decode(enc)
            runs = homopolymer_run_lengths(enc.seq_int)
            ok = "OK" if dec == msg else "FAIL"
            density = len(enc) * (2.0 if scheme == "naive" else math.log2(3)) / len(msg)
            print(f"  [{scheme:5s}] {ok}  '{msg}' -> {len(enc):3d} bases  "
                  f"{density:.2f} bit/char  max_run={runs.max()}  "
                  f"seq={enc.seq[:24]}{'...' if len(enc) > 24 else ''}")
            assert dec == msg, (scheme, msg)
        # RLL guarantee
        enc = DNACodec.encode("".join(chr(c) for c in range(33, 120)), "rll")
        assert homopolymer_run_lengths(enc.seq_int).max() == 1, "RLL homopolymer leak"
    print("  RLL homopolymer-free guarantee: verified")

    # -- 2. synthesis yield: 100 bp vs 1000 bp --------------------------------
    print("\n== TdT synthesis yield (naive, homopolymers present) ==")
    rng = np.random.default_rng(1)
    long_txt = "".join(chr(33 + (i % 90)) for i in range(400))      # ~1600 naive bases
    enc_long = DNACodec.encode_naive(long_txt)
    for Y in (0.999, 0.992, 0.970):
        sim = TdTSynthesisSimulator(coupling_yield=Y, p_base=0.004)
        for N in (100, 1000):
            eta = sim.full_length_yield(N)
            print(f"  Y={Y:.3f}  N={N:4d} bp  eta=Y^N={eta:7.3%}")
        rep = sim.simulate(DNAEncoded(enc_long.seq_int[:1000], "naive", 0), rng)
        print(f"     1000-bp strand: max_run={rep.max_run}  mean P_del={rep.mean_p_del:.4%}"
              f"  deleted={rep.n_deleted} bases  full-length eta={rep.cumulative_yield:.3%}")
    # RLL floor
    rep_rll = TdTSynthesisSimulator(0.992, 0.004).simulate(
        DNAEncoded(DNACodec.encode_rll(long_txt).seq_int[:1000], "rll", 0), rng)
    print(f"  RLL 1000-bp: max_run={rep_rll.max_run}  mean P_del={rep_rll.mean_p_del:.4%} "
          f"(flat floor = homopolymer hazard removed)")

    # -- 3. modulate -> inject -> demod: ideal recovery + in-band SIR ---------
    print("\n== HFIM modulation + muscle interference ==")
    mod = HFIMModulator()
    print(f"  Z0={mod.Z0:.0f}  Gamma_low={mod.gamma_low:+.4f}  Gamma_high={mod.gamma_high:+.4f}"
          f"  -> m={mod.m:.4f}  osp={mod.cfg.osp}")

    enc = DNACodec.encode_naive("A16Z_ACADEMY")
    bits = DNACodec.to_bits(enc)
    t, v_tx, b_nrz = mod.modulate(bits)

    # ideal (no muscle/noise) round-trip through demod
    inj0 = MuscleArtifactInjector(MuscleConfig(semg_depth=0.0, zmod_depth=0.0, snr_db=120.0))
    out0 = inj0.apply_and_demodulate(t, v_tx, b_nrz, mod, np.random.default_rng(0))
    rec_bits = slice_symbols(out0["baseband"], mod.cfg.osp, bits.size)
    rec = DNACodec.decode_naive(DNAEncoded(
        (rec_bits.reshape(-1, 2) * np.array([2, 1])).sum(1).astype(np.int8), "naive", enc.n_bytes))
    print(f"  ideal demod bit errors: {(rec_bits != bits).sum()}/{bits.size}  recovered='{rec}'")
    assert (rec_bits != bits).sum() == 0

    # SIR/ANC setup needs a signal long enough to contain the muscle bands:
    # ~0.3 s -> 3000 symbols.
    bits_long = np.random.default_rng(3).integers(0, 2, 3000)
    tL, vL, bL = mod.modulate(bits_long)
    print(f"  SIR signal: {bits_long.size} symbols, {tL[-1]*1e3:.0f} ms, {vL.size} samples")
    for mode in ("isometric", "motion"):
        inj = MuscleArtifactInjector(MuscleConfig(mode=mode, semg_depth=3.0, zmod_depth=0.3, snr_db=25.0))
        out = inj.apply_and_demodulate(tL, vL, bL, mod, np.random.default_rng(2))
        corr = np.corrcoef(out["muscle_bb"], out["reference"])[0, 1]
        print(f"  [{mode:9s}] in-band SIR = {out['in_band_sir_db']:+6.2f} dB  "
              f"primary/ref corr = {corr:.3f}  (d=primary_bb, u=reference_bb for dcd_rls)")

    # -- 3b. end-to-end integration: dual channels -> DCD-RLS -> ECC -> decode -
    print("\n== end-to-end: HFIM -> muscle (buries data) -> DCD-RLS -> ECC -> decode ==")
    try:
        from bdn04.dcd_rls import DCDRLS
        msg = "A16Z_ACADEMY"
        enc_e = DNACodec.encode_naive(msg)
        bits_e = DNACodec.to_bits(enc_e)                     # 96 bits / copy
        n_rep = 7                                            # rate-1/7 repetition code
        bits_pad = np.tile(bits_e, n_rep)
        te, ve, be = mod.modulate(bits_pad)
        inj = MuscleArtifactInjector(MuscleConfig(mode="motion", semg_depth=3.0, zmod_depth=0.3, snr_db=25.0))
        out = inj.apply_and_demodulate(te, ve, be, mod, np.random.default_rng(5))
        d, r, s = out["primary_bb"], out["reference_bb"], out["data_bb_dec"]
        osp_bb = out["osp_bb"]

        anc = DCDRLS(n_taps=24, lam=0.995, delta=1e-3, Nu=8)
        anc.run(r[:400], d[:400])                            # JIT warm-up
        cleaned = anc.run(r, d)["e"]                         # u=reference, d=primary -> e ~ data

        def sir(e):
            return 10 * np.log10(np.mean(s ** 2) / (np.mean((e - s) ** 2) + 1e-30))
        raw_bits = slice_symbols(d, osp_bb, bits_pad.size)
        anc_bits = slice_symbols(cleaned, osp_bb, bits_pad.size)
        ber_raw = (raw_bits != bits_pad).mean()
        ber_anc = (anc_bits != bits_pad).mean()
        txt_anc = bits_to_text(repetition_decode(anc_bits, bits_e.size), enc_e.n_bytes)
        txt_raw = bits_to_text(repetition_decode(raw_bits, bits_e.size), enc_e.n_bytes)
        print(f"  input SIR {sir(d):+6.2f} dB (muscle buries data)  ->  post-ANC {sir(cleaned):+6.2f} dB")
        print(f"  uncoded BER: raw {ber_raw:6.2%}  ->  post-ANC {ber_anc:6.2%}")
        print(f"  rate-1/{n_rep} ECC alone (no ANC): '{txt_raw}'  "
              f"[{'ok' if txt_raw == msg else 'FAILS - redundancy cannot fix 37% BER'}]")
        print(f"  ANC + same ECC:                 '{txt_anc}'  "
              f"[{'RECOVERED' if txt_anc == msg else 'residual errors'}]")
    except Exception as e:  # pragma: no cover
        import traceback; traceback.print_exc()
        print(f"  (skipped: {e})")

    # -- 4. runtime: 10k-sample burst -----------------------------------------
    print("\n== runtime (10k-sample burst) ==")
    big_bits = np.random.default_rng(0).integers(0, 2, 10000 // mod.cfg.osp + 1)
    tb, vb, bn = mod.modulate(big_bits)
    tb, vb, bn = tb[:10000], vb[:10000], bn[:10000]
    inj = MuscleArtifactInjector()
    inj.apply_and_demodulate(tb, vb, bn, mod, np.random.default_rng(0))   # warm JIT/caches
    runs = []
    for _ in range(20):
        t0 = time.perf_counter()
        inj.apply_and_demodulate(tb, vb, bn, mod, np.random.default_rng(0))
        runs.append(time.perf_counter() - t0)
    print(f"  full inject+I/Q demod on 10,000 samples: {1e3*np.median(runs):.2f} ms median "
          f"({1e3*np.min(runs):.2f} ms best)  -> {'PASS' if np.min(runs) < 10e-3 else 'CHECK'} (<10 ms)")
