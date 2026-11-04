"""
BDN-04 :: Outer error-correcting code (Reed-Solomon over GF(256))
=================================================================

Block-based RS is how real DNA data storage ships (Grass 2015 used RS; Erlich-
Zielinski 2017 used a fountain code). We model RS(255, 255-nsym): the payload is
split into k-byte data blocks, each extended with nsym parity bytes, correcting
up to t = nsym/2 *byte* errors per block. Because RS corrects whole-byte symbols,
a few wrong bits inside a byte cost the same as one, so clustered post-ANC errors
are cheap and random high-BER noise is catastrophic -- exactly the regime that
separates "ECC alone" from "ANC + ECC".

Bit ordering matches dna_hfim.DNACodec.to_bits: MSB-first, 2 bits/base.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from reedsolo import RSCodec, ReedSolomonError


@dataclass
class ECCResult:
    ok: bool
    text: str
    n_byte_errors: int           # corrupted bytes presented to the decoder
    t_capacity: int              # correction capacity t = nsym/2 per block
    corrected: int               # byte errors the decoder actually fixed (-1 if failed)


class ReedSolomonECC:
    def __init__(self, nsym: int = 32, nsize: int = 255):
        """nsym parity bytes per block => corrects floor(nsym/2) byte errors/block."""
        self.nsym = nsym
        self.nsize = nsize
        self.rsc = RSCodec(nsym, nsize=nsize)
        self.t = nsym // 2

    # ---- byte <-> bit helpers (MSB-first, matches DNACodec.to_bits) --------- #
    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        n = (bits.size // 8) * 8
        return np.packbits(bits[:n].astype(np.uint8)).tobytes()

    # ---- encode: text -> RS-coded bytes -> bits ----------------------------- #
    def encode_text(self, text: str) -> tuple[bytes, np.ndarray]:
        coded = bytes(self.rsc.encode(text.encode("utf-8")))
        return coded, self.bytes_to_bits(coded)

    def code_rate(self, n_payload_bytes: int) -> float:
        k = self.nsize - self.nsym
        n_blocks = int(np.ceil(n_payload_bytes / k))
        coded_len = n_payload_bytes + n_blocks * self.nsym
        return n_payload_bytes / coded_len

    # ---- decode: recovered bits -> text, with honest failure flag ----------- #
    def decode_bits(self, bits: np.ndarray, coded_ref: bytes) -> ECCResult:
        recv = bytearray(self.bits_to_bytes(bits))
        ref = bytearray(coded_ref)
        L = min(len(recv), len(ref))
        recv = recv[:L]
        n_err = int(sum(1 for a, b in zip(recv, ref) if a != b))
        try:
            decoded, _, errata = self.rsc.decode(bytes(recv))
            text = bytes(decoded).decode("utf-8", errors="replace")
            return ECCResult(True, text, n_err, self.t, len(errata))
        except (ReedSolomonError, UnicodeError, Exception):
            return ECCResult(False, "", n_err, self.t, -1)


if __name__ == "__main__":
    import time
    from bdn04.dna_hfim import (HFIMModulator, MuscleConfig, MuscleArtifactInjector,
                                slice_symbols)
    from bdn04.dcd_rls import DCDRLS

    print("== RS(255, 255-nsym) capacity sanity ==")
    rng = np.random.default_rng(0)
    for nsym in (16, 32, 48):
        ecc = ReedSolomonECC(nsym=nsym)
        coded, bits = ecc.encode_text("A16Z_ACADEMY")
        print(f"  nsym={nsym:2d} t={ecc.t:2d}  'A16Z_ACADEMY'(12B) -> {len(coded)}B coded, "
              f"rate={ecc.code_rate(12):.3f}, {bits.size} bits")
        for n_flip in (ecc.t, ecc.t + 1):                       # at capacity, then one past
            b = bytearray(coded)
            idx = rng.choice(len(b), n_flip, replace=False)
            for i in idx:
                b[i] ^= 0xFF
            res = ecc.decode_bits(ecc.bytes_to_bits(bytes(b)), coded)
            print(f"     {n_flip:2d} byte errors -> {'OK' if res.ok else 'FAIL':4s} "
                  f"(corrected {res.corrected})")

    # ---- the demo contrast on the real pipeline ---------------------------- #
    print("\n== pipeline contrast: RS alone vs ANC + RS ==")
    mod = HFIMModulator()
    ecc = ReedSolomonECC(nsym=40)
    coded, bits = ecc.encode_text("A16Z_ACADEMY")
    print(f"  payload 12B -> RS(nsym=40) {len(coded)}B, t={ecc.t}, rate {ecc.code_rate(12):.3f}, "
          f"{bits.size} coded bits")
    okA = okR = 0
    for seed in range(8):
        t, v, b = mod.modulate(bits)
        inj = MuscleArtifactInjector(MuscleConfig(mode="motion", semg_depth=3.0, zmod_depth=0.3, snr_db=25.0))
        out = inj.apply_and_demodulate(t, v, b, mod, np.random.default_rng(seed))
        d, r = out["primary_bb"], out["reference_bb"]
        ob = out["osp_bb"]
        anc = DCDRLS(n_taps=24, lam=0.995, delta=1e-3, Nu=8)
        anc.run(r[:400], d[:400])
        e = anc.run(r, d)["e"]
        raw = ecc.decode_bits(slice_symbols(d, ob, bits.size), coded)
        rec = ecc.decode_bits(slice_symbols(e, ob, bits.size), coded)
        okR += raw.ok and raw.text == "A16Z_ACADEMY"
        okA += rec.ok and rec.text == "A16Z_ACADEMY"
        if seed == 0:
            print(f"  seed0: RS-alone {raw.n_byte_errors:3d} byte errs -> {'OK' if raw.ok else 'FAIL'} ; "
                  f"ANC+RS {rec.n_byte_errors:3d} byte errs -> {'OK' if rec.ok else 'FAIL'} '{rec.text}'")
    print(f"  over 8 seeds:  RS alone recovers {okR}/8   |   ANC + RS recovers {okA}/8")
