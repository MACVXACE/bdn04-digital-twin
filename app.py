"""
BDN-04 :: Wetware-to-Edge Digital Twin
======================================
High-density scientific-instrument dashboard for the BDN-04 architecture:
enzymatic DNA data storage -> EQS-HBC body channel -> biological interference ->
DCD-RLS adaptive recovery -> Reed-Solomon decode.

    streamlit run app.py

All physics/DSP lives in bdn04/*.py; this file is pure UI + Plotly wiring on top
of bdn04.pipeline (which is import-testable without Streamlit).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from bdn04.pipeline import run_storage, run_channel, run_link, LinkParams
from bdn04.tissue import PATHS

# --------------------------------------------------------------------------- #
# palette / theme
# --------------------------------------------------------------------------- #
BG      = "#0e1117"
PANEL   = "#141922"
GRID    = "#222a38"
INK     = "#c9d3e0"
MUTED   = "#64748b"
CLEAN   = "#22d3ee"   # cyan   - clean signal
CORRUPT = "#f43f5e"   # rose   - interference
RECOVER = "#4ade80"   # green  - recovered
AMBER   = "#f59e0b"
ACCENT  = "#7dd3fc"
MONO    = "'JetBrains Mono','SF Mono',Menlo,Consolas,monospace"

st.set_page_config(page_title="BDN-04 Digital Twin", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; color:{INK}; }}
  #MainMenu, header, footer {{ visibility:hidden; }}
  .block-container {{ padding:0.6rem 1.1rem 2rem 1.1rem; max-width:100%; }}
  html, body, [class*="css"], .stMarkdown, .stMetric {{ font-family:{MONO}; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {GRID}; }}
  section[data-testid="stSidebar"] * {{ font-family:{MONO}; }}
  .hbar {{ background:linear-gradient(90deg,{PANEL},{BG}); border:1px solid {GRID};
           border-left:3px solid {ACCENT}; border-radius:4px; padding:5px 10px;
           margin:10px 0 6px 0; font-size:0.74rem; letter-spacing:.14em;
           color:{ACCENT}; text-transform:uppercase; }}
  .card {{ background:{PANEL}; border:1px solid {GRID}; border-radius:5px;
           padding:7px 10px; height:100%; }}
  .card .lab {{ color:{MUTED}; font-size:0.60rem; letter-spacing:.11em;
                text-transform:uppercase; }}
  .card .val {{ font-size:1.28rem; font-weight:600; line-height:1.25; }}
  .card .sub {{ color:{MUTED}; font-size:0.62rem; }}
  .seqbox {{ background:#0b0e14; border:1px solid {GRID}; border-radius:4px;
             padding:7px 9px; font-size:0.72rem; word-break:break-all; color:{INK}; }}
  .pill {{ display:inline-block; padding:1px 7px; border-radius:9px;
           font-size:0.62rem; letter-spacing:.08em; }}
  .stButton>button {{ font-family:{MONO}; font-weight:600; letter-spacing:.08em;
           background:{CORRUPT}; color:#140406; border:1px solid {CORRUPT};
           border-radius:5px; width:100%; }}
  .stButton>button:hover {{ background:#ff5c74; border-color:#ff5c74; color:#140406; }}
  div[data-baseweb="select"]>div, .stTextInput input {{ font-family:{MONO}; }}
  .title {{ font-size:1.18rem; font-weight:700; letter-spacing:.04em; color:#e8eef7; }}
  .sublt {{ color:{MUTED}; font-size:0.66rem; letter-spacing:.06em; }}
</style>
""", unsafe_allow_html=True)


def card(label, value, color=INK, sub=""):
    return (f"<div class='card'><div class='lab'>{label}</div>"
            f"<div class='val' style='color:{color}'>{value}</div>"
            f"<div class='sub'>{sub}</div></div>")


def hbar(text):
    st.markdown(f"<div class='hbar'>{text}</div>", unsafe_allow_html=True)


def style(fig, h, ytitle=""):
    fig.update_layout(template="plotly_dark", height=h,
                      paper_bgcolor=BG, plot_bgcolor="#0b0e14",
                      margin=dict(l=54, r=14, t=24, b=30), showlegend=True,
                      legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
                      font=dict(family=MONO, size=11, color=INK))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=ytitle, title_font=dict(size=10))
    return fig


# --------------------------------------------------------------------------- #
# cached pipeline wrappers (primitive args -> hashable)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def c_storage(text, N, Y, pb):
    return run_storage(text, N, Y, pb)

@st.cache_data(show_spinner=False)
def c_channel(path, dist, load):
    return run_channel(path, dist, load)

@st.cache_data(show_spinner=False)
def c_link(text, nsym, mode, semg, zmod, snr, flex, taps, lam, Nu, seed):
    return run_link(LinkParams(text=text, nsym=nsym, mode=mode, semg_depth=semg,
                               zmod_depth=zmod, snr_db=snr, flex=flex, taps=taps,
                               lam=lam, Nu=Nu, seed=seed))


# =========================================================================== #
# sidebar controls
# =========================================================================== #
S = st.sidebar
S.markdown(f"<div class='title'>BDN-04</div>"
           f"<div class='sublt'>WETWARE-TO-EDGE DIGITAL TWIN</div>"
           f"<div class='sublt'>EQS-HBC · DOI 10.5281/zenodo.19699441</div><br>",
           unsafe_allow_html=True)

S.markdown("**◢ STORAGE**")
payload = S.text_input("PAYLOAD", "A16Z_ACADEMY", max_chars=40)
N_synth = S.slider("TdT SEQUENCE LENGTH (bp)", 50, 2000, 500, 50)
Y = S.slider("STEPWISE COUPLING YIELD", 0.970, 0.999, 0.992, 0.001, format="%.3f")
p_base = S.slider("HOMOPOLYMER HAZARD p_base", 0.000, 0.020, 0.004, 0.001, format="%.3f")

S.markdown("**◢ CHANNEL (EQS-HBC)**")
path = S.selectbox("TRANSMISSION PATH", list(PATHS.keys()), index=0,
                   format_func=lambda s: s.replace("_", " → "))
from bdn04.tissue import path_length
dist = S.slider("TX→RX DISTANCE (m)", 0.10, float(round(path_length(path), 2)),
                min(0.9, float(round(path_length(path), 2))), 0.05)
load_hi = S.toggle("HIGH-Z RECEIVER (off = 50 Ω)", value=True)
load_mode = "high_z" if load_hi else "50ohm"

S.markdown("**◢ INTERFERENCE & RECOVERY**")
mode = S.selectbox("MUSCLE MODE", ["motion", "isometric"])
semg = S.slider("sEMG DEPTH (×data RMS)", 0.0, 6.0, 3.0, 0.1)
zmod = S.slider("ΔZ ENVELOPE DEPTH", 0.0, 1.0, 0.3, 0.05)
snr = S.slider("CHANNEL SNR (dB)", 5, 40, 25, 1)
nsym = S.slider("REED-SOLOMON PARITY (nsym)", 8, 80, 40, 2)
with S.expander("DCD-RLS filter"):
    taps = st.slider("TAPS (N)", 8, 48, 24, 2)
    lam = st.slider("FORGETTING λ", 0.980, 0.9999, 0.995, 0.001, format="%.3f")
    Nu = st.slider("DCD UPDATES / SAMPLE Nu", 1, 16, 8, 1)
    seed = st.slider("NOISE SEED", 0, 50, 5, 1)

# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
    f"border-bottom:1px solid {GRID};padding-bottom:6px;'>"
    f"<span class='title'>WETWARE-TO-EDGE DIGITAL TWIN</span>"
    f"<span class='sublt'>human body as electroquasistatic volume conductor · "
    f"500 kHz HFIM · DCD-RLS · RS(255,k)</span></div>", unsafe_allow_html=True)

sto = c_storage(payload, N_synth, Y, p_base)

# =========================================================================== #
# SECTION 1 : STORAGE
# =========================================================================== #
hbar("01 · Enzymatic DNA Storage")
c = st.columns([1, 1, 1, 1])
c[0].markdown(card("η(N) FULL-LENGTH YIELD", f"{sto['eta_N']*100:.2f}%",
                   RECOVER if sto['eta_N'] > 0.3 else AMBER if sto['eta_N'] > 0.02 else CORRUPT,
                   f"Y^N, N={sto['N_synth']} bp @ Y={sto['coupling_yield']:.3f}"), unsafe_allow_html=True)
c[1].markdown(card("HOMOPOLYMER HAZARD", f"{sto['mean_p_del_naive']*100:.3f}%",
                   AMBER, f"mean P_del · naive, max run {sto['max_run_naive']}"), unsafe_allow_html=True)
c[2].markdown(card("NAIVE CODE", f"{sto['len_naive']} nt",
                   CLEAN, f"{sto['bits_per_char_naive']:.2f} bit/char · runs≤{sto['max_run_naive']}"),
              unsafe_allow_html=True)
c[3].markdown(card("RLL / GOLDMAN CODE", f"{sto['len_rll']} nt",
                   RECOVER, f"{sto['bits_per_char_rll']:.2f} bit/char · runs≤{sto['max_run_rll']} (no homopolymers)"),
              unsafe_allow_html=True)

cc = st.columns([3, 2])
cc[0].markdown("<div class='sub' style='color:#64748b;font-size:.62rem'>NAIVE ATCG (homopolymers present)</div>"
               f"<div class='seqbox'>{sto['dna_naive']}</div>"
               "<div class='sub' style='color:#64748b;font-size:.62rem;margin-top:5px'>RLL ATCG (run-length constrained)</div>"
               f"<div class='seqbox'>{sto['dna_rll']}</div>", unsafe_allow_html=True)
# yield decay curve vs length
Ngrid = np.arange(10, 2001, 10)
fig = go.Figure()
for yv, col in ((0.999, RECOVER), (sto['coupling_yield'], CLEAN), (0.970, CORRUPT)):
    fig.add_scatter(x=Ngrid, y=yv ** Ngrid * 100, mode="lines",
                    line=dict(color=col, width=2), name=f"Y={yv:.3f}")
fig.add_vline(x=sto['N_synth'], line=dict(color=MUTED, dash="dot"))
fig.update_yaxes(range=[0, 100])
cc[1].plotly_chart(style(fig, 210, "η(N) full-length  [%]").update_xaxes(title_text="synthesis length N  [bp]"),
                   width='stretch', key="yield")

# =========================================================================== #
# SECTION 2 : CHANNEL
# =========================================================================== #
hbar("02 · EQS-HBC Physical Channel  ·  4-pole Cole-Cole tissue")
ch = c_channel(path, dist, load_mode)
cc = st.columns([3, 1])
bode = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                     row_heights=[0.56, 0.44])
bode.add_scatter(x=ch["f"], y=ch["mag_db"], line=dict(color=CLEAN, width=2),
                 name="|H| magnitude", row=1, col=1)
bode.add_scatter(x=ch["f"], y=ch["phase_deg"], line=dict(color=AMBER, width=2),
                 name="phase", row=2, col=1)
bode.add_vline(x=ch["f_mark"], line=dict(color=MUTED, dash="dot"))
bode.add_annotation(x=np.log10(ch["f_mark"]), y=ch["mag_at_mark"], row=1, col=1,
                    text=f" 500 kHz: {ch['loss_500k_db']:.1f} dB", showarrow=True,
                    arrowcolor=ACCENT, font=dict(color=ACCENT, size=11), ax=42, ay=-22)
bode.update_xaxes(type="log", gridcolor=GRID)
bode.update_xaxes(title_text="frequency  [Hz]", row=2, col=1, title_font=dict(size=10))
bode.update_yaxes(title_text="|H|  [dB]", row=1, col=1, gridcolor=GRID, title_font=dict(size=10))
bode.update_yaxes(title_text="∠H  [deg]", row=2, col=1, gridcolor=GRID, title_font=dict(size=10))
cc[0].plotly_chart(style(bode, 330), width='stretch', key="bode")

ld = "HIGH-Z" if load_mode == "high_z" else "50 Ω"
lcol = CLEAN if load_mode == "high_z" else CORRUPT
cc[1].markdown(card("CHANNEL LOSS @ 500 kHz", f"{ch['loss_500k_db']:.1f} dB", lcol,
                    f"{ld} termination"), unsafe_allow_html=True)
cc[1].markdown(card("TISSUE SERIES Z", f"{ch['z_tissue_ohm']:.0f} Ω",
                    INK, f"∠{ch['z_tissue_phase']:.1f}° · d={ch['distance']:.2f} m"), unsafe_allow_html=True)
if load_mode == "high_z":
    cc[1].markdown(card("DISTANCE DEPENDENCE", "≈ NONE", RECOVER,
                        "return-path limited, not tissue — EQS invariance"), unsafe_allow_html=True)
else:
    cc[1].markdown(card("50 Ω TRAP", "HIGH-PASS", CORRUPT,
                        "loading the body kills the LF signal"), unsafe_allow_html=True)

# =========================================================================== #
# SECTION 3 : OSCILLOSCOPE
# =========================================================================== #
if "flex" not in st.session_state:
    st.session_state.flex = False
hh = st.columns([3, 1])
hh[0].markdown(f"<div class='hbar'>03 · Live Oscilloscope  ·  HFIM → sEMG → DCD-RLS</div>",
               unsafe_allow_html=True)
if hh[1].button("⚡ FLEX MUSCLE" + ("  ▣ ON" if st.session_state.flex else "")):
    st.session_state.flex = not st.session_state.flex

L = c_link(payload, nsym, mode, semg, zmod, snr, st.session_state.flex, taps, lam, Nu, seed)

m = st.columns(4)
m[0].markdown(card("SIR IN → OUT", f"{L['sir_in']:+.1f} → {L['sir_out']:+.1f} dB",
                   RECOVER, f"ANC gain {L['sir_out']-L['sir_in']:+.1f} dB"), unsafe_allow_html=True)
m[1].markdown(card("UNCODED BER", f"{L['ber_raw']*100:.1f}% → {L['ber_anc']*100:.1f}%",
                   RECOVER if L['ber_anc'] < L['ber_raw'] else CORRUPT, "raw → post-ANC"), unsafe_allow_html=True)
m[2].markdown(card("DCD-RLS LOCK", f"{L['lock_ms']:.2f} ms" if L['lock_ms'] >= 0 else "—",
                   CLEAN, f"{L['n_updates_mean']:.1f} updates/sample · N={taps}"), unsafe_allow_html=True)
m[3].markdown(card("sEMG STATE", "⚡ FLEXED" if st.session_state.flex else "RESTING",
                   CORRUPT if st.session_state.flex else MUTED,
                   f"depth {L['semg_depth_eff']:.1f}× · {mode}"), unsafe_allow_html=True)

# scope 1 : RF carrier
f1 = go.Figure()
f1.add_scatter(x=L["rf_t_us"], y=L["rf_clean"], line=dict(color=CLEAN, width=1.3),
               name="clean HFIM carrier (500 kHz)")
f1.update_xaxes(title_text="time  [µs]", title_font=dict(size=10))
st.plotly_chart(style(f1, 190, "v_tx  [a.u.]"), width='stretch', key="sc1")

# scope 2 : baseband buried
f2 = go.Figure()
f2.add_scatter(x=L["bb_t_ms"], y=L["bb_primary"], line=dict(color=CORRUPT, width=1.0),
               name="received (data + sEMG)")
f2.add_scatter(x=L["bb_t_ms"], y=L["bb_data"], line=dict(color=INK, width=1.3, dash="dot"),
               name="true data baseband")
f2.update_xaxes(title_text="time  [ms]", title_font=dict(size=10))
st.plotly_chart(style(f2, 190, "baseband  [a.u.]"), width='stretch', key="sc2")

# scope 3 : ANC recovery
f3 = go.Figure()
f3.add_scatter(x=L["bb_t_ms"], y=L["bb_anc"], line=dict(color=RECOVER, width=1.3),
               name="DCD-RLS output e(n)")
f3.add_scatter(x=L["bb_t_ms"], y=L["bb_data"], line=dict(color=INK, width=1.0, dash="dot"),
               name="true data baseband")
if L["lock_ms"] >= 0:
    f3.add_vline(x=L["lock_ms"], line=dict(color=AMBER, dash="dot"))
f3.update_xaxes(title_text="time  [ms]", title_font=dict(size=10))
st.plotly_chart(style(f3, 190, "recovered  [a.u.]"), width='stretch', key="sc3")

# =========================================================================== #
# SECTION 4 : RECOVERY
# =========================================================================== #
hbar("04 · Payload Recovery  ·  Reed-Solomon over GF(256)")
r = st.columns([1, 1, 1])
r[0].markdown(card("RS CODE", f"RS(255, {255-nsym})",
                   INK, f"{L['payload_bytes']}B → {L['coded_len']}B · rate {L['code_rate']:.2f} · t={L['t_capacity']} B"),
              unsafe_allow_html=True)

raw_c = RECOVER if L["raw_ok"] else CORRUPT
anc_c = RECOVER if L["anc_ok"] else CORRUPT
r[1].markdown(
    f"<div class='card' style='border-left:3px solid {raw_c}'>"
    f"<div class='lab'>RS ALONE &nbsp;(no adaptive filter)</div>"
    f"<div class='val' style='color:{raw_c};font-size:1.0rem'>"
    f"{(L['raw_text'] or '∅')[:40]}</div>"
    f"<div class='sub'>{L['raw_byte_err']} corrupted bytes vs t={L['t_capacity']} → "
    f"<span style='color:{raw_c}'>{'RECOVERED' if L['raw_ok'] else 'ECC FAILS'}</span></div></div>",
    unsafe_allow_html=True)
r[2].markdown(
    f"<div class='card' style='border-left:3px solid {anc_c}'>"
    f"<div class='lab'>DCD-RLS + RS</div>"
    f"<div class='val' style='color:{anc_c};font-size:1.0rem'>"
    f"{(L['anc_text'] or '∅')[:40]}</div>"
    f"<div class='sub'>{L['anc_byte_err']} corrupted bytes vs t={L['t_capacity']} → "
    f"<span style='color:{anc_c}'>{'RECOVERED' if L['anc_ok'] else 'residual errors'}</span></div></div>",
    unsafe_allow_html=True)

if L["anc_ok"] and not L["raw_ok"]:
    st.markdown(f"<div style='text-align:center;color:{RECOVER};font-size:0.8rem;"
                f"letter-spacing:.1em;margin-top:8px'>▣ THESIS CONFIRMED — "
                f"identical ECC budget: muscle interference defeats RS alone; "
                f"DCD-RLS cancellation restores the payload.</div>", unsafe_allow_html=True)

st.markdown(f"<div class='sublt' style='margin-top:10px'>DNA on the wire (RS-coded, "
            f"{len(L['dna_tx'])} nt): </div><div class='seqbox' style='font-size:.62rem'>{L['dna_tx']}</div>",
            unsafe_allow_html=True)
