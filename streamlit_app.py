import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from io import BytesIO

# --------- Chargement séries (cache) ---------
@st.cache_data(show_spinner=False)
def load_series():
    y_test = pd.read_csv("y_test.csv", index_col=0, parse_dates=True).iloc[:, 0]
    yhat_aic = pd.read_csv("yhat_aligned_aic_bic.csv", index_col=0, parse_dates=True).iloc[:, 0]
    yhat_rmse = pd.read_csv("yhat_aligned_rmse.csv", index_col=0, parse_dates=True).iloc[:, 0]
    yhat_aic = yhat_aic.reindex(y_test.index)
    yhat_rmse = yhat_rmse.reindex(y_test.index)
    return y_test, yhat_aic, yhat_rmse

y_test, yhat_aligned_aic_bic, yhat_aligned_rmse = load_series()

# --------- Streamlit : plein écran sans UI ---------
st.set_page_config(layout="wide", page_title="STREAMING", initial_sidebar_state="collapsed")
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {padding: 0; margin: 0;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ======== Réglages fluidité ========
TARGET_FPS   = 1000                 # Hz
REDRAW_EVERY = 1.0 / TARGET_FPS   # s
WINDOW       = 1000               # points visibles
MAX_PTS      = 1000               # cap visuel
SMOOTH_SPAN  = 3                  # EMA (0/1 pour off)
JPEG_QUALITY = 70                 # qualité JPEG (taille vs netteté)
DPI          = 90                 # DPI bas -> plus léger
LINE_SCALE   = 3.0                # facteur pour épaissir les courbes

# --------- Pré-calculs ---------
idx = y_test.index
# 1) Dates numérisées UNE FOIS (float) pour Matplotlib
x_full_num = mdates.date2num(idx.to_pydatetime()).astype(np.float64)

# 2) Lissage en amont (une fois) + cast float32 pour réduire le volume
def smooth_ema(s: pd.Series, span: int):
    if span and span > 1:
        return s.ewm(span=span, adjust=False).mean().astype(np.float32).values
    return s.astype(np.float32).values

y_true_all = smooth_ema(y_test, SMOOTH_SPAN)
y_aic_all  = smooth_ema(yhat_aligned_aic_bic, SMOOTH_SPAN)
y_rms_all  = smooth_ema(yhat_aligned_rmse, SMOOTH_SPAN)

valid_aic_mask  = ~np.isnan(y_aic_all)
valid_rms_mask  = ~np.isnan(y_rms_all)

# --------- Figure persistante (une seule fois) ---------
fig, ax = plt.subplots(figsize=(22, 10), dpi=DPI)

# Fond noir
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.grid(alpha=0.25)

# Labels + ticks en blanc
ax.set_xlabel("Time", color="white")
ax.set_ylabel("Value", color="white")
ax.tick_params(colors="white")

# Axe X en mode date
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))

# Titre multilignes en blanc (ligne 1 fixe, ligne 2 sera mise à jour dans la boucle)
ax.set_title(
    "Internet traffic forecasting\nRMSE AIC=... · RMSE Backtesting=...",
    color="white",
    fontsize=18
)

(line_true,) = ax.plot([], [], linewidth=1.8 * LINE_SCALE,
                       label="Internet traffic", color="#87CEEB")
(line_aic,)  = ax.plot([], [], linewidth=2.0 * LINE_SCALE, linestyle="--",
                       label="ARIMA AIC", color="red")
(line_rms,)  = ax.plot([], [], linewidth=2.0 * LINE_SCALE, linestyle="--",
                       label="ARIMA Backtesting", color="orange")

vline = ax.axvline(
    x_full_num[0],
    linestyle="--",
    linewidth=1.0 * LINE_SCALE,
    alpha=0.5,
    color="#666666"
)

# Agrandissement de la légende ×3
base_size = plt.rcParams.get("legend.fontsize", 10)
if isinstance(base_size, str):
    # Valeur numérique par défaut si rcParams renvoie un mot-clé ("medium", etc.)
    base_size = 10
leg = ax.legend(loc="upper left", fontsize=3 * base_size)

# Style de la légende
leg.get_frame().set_facecolor("black")
leg.get_frame().set_edgecolor("white")
for t in leg.get_texts():
    t.set_color("white")

# Placeholders séparés : texte (RMSE) et image (figure)
ph_text = st.empty()
ph_fig  = st.empty()

# --------- Helpers ultra-légers ---------
def window_slice(i):
    start = max(0, i - WINDOW)
    stop  = i
    return start, stop

def downsample_indices_fast(n: int, max_pts: int):
    """Indice régulier via slicing O(1), en garantissant l'index final."""
    if n <= max_pts:
        return None  # sentinel -> pas de downsample
    step = n // max_pts  # >= 1
    # on veut toujours inclure le dernier point
    return step

# RMSE incrémental
sse_aic = 0.0
cnt_aic = 0
sse_rms = 0.0
cnt_rms = 0

last_draw = 0.0
t0 = time.time()

N = len(x_full_num)

for i in range(1, N + 1):
    k = i - 1
    # Maj RMSE incrémental
    ya = y_aic_all[k]
    yr = y_rms_all[k]
    yt = y_true_all[k]

    if valid_aic_mask[k]:
        d = yt - ya
        if d == d:  # nan-check rapide
            sse_aic += float(d * d)
            cnt_aic += 1

    if valid_rms_mask[k]:
        d = yt - yr
        if d == d:
            sse_rms += float(d * d)
            cnt_rms += 1

    # Gate de rafraîchissement
    now = time.time()
    if (now - last_draw) < REDRAW_EVERY and i < N:
        continue
    last_draw = now

    # Fenêtre
    start, stop = window_slice(i)
    x_win  = x_full_num[start:stop]
    y_twin = y_true_all[start:stop]
    y_awin = y_aic_all[start:stop]
    y_rwin = y_rms_all[start:stop]

    # Downsample rapide
    n = stop - start
    step = downsample_indices_fast(n, MAX_PTS)
    if step is None:
        x_sel, y_ts, y_as, y_rs = x_win, y_twin, y_awin, y_rwin
    else:
        # Slicing régulier + inclusion du dernier point
        x_sel  = x_win[::step]
        y_ts   = y_twin[::step]
        y_as   = y_awin[::step]
        y_rs   = y_rwin[::step]
        if x_sel[-1] != x_win[-1]:
            # concatène le dernier point (coût négligeable)
            x_sel  = np.concatenate((x_sel,  x_win[-1:]))
            y_ts   = np.concatenate((y_ts,   y_twin[-1:]))
            y_as   = np.concatenate((y_as,   y_awin[-1:]))
            y_rs   = np.concatenate((y_rs,   y_rwin[-1:]))

    # Mettre à jour les artistes
    line_true.set_data(x_sel, y_ts)
    line_aic.set_data(x_sel, y_as)
    line_rms.set_data(x_sel, y_rs)

    # Limites axes
    ax.set_xlim(x_sel[0], x_sel[-1])
    # y-lims : une passe
    stack = np.vstack((y_ts, y_as, y_rs))
    ymin = np.nanmin(stack)
    ymax = np.nanmax(stack)
    pad  = (ymax - ymin) * 0.05 if np.isfinite(ymax - ymin) else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    # ---- Rendu JPEG léger -> st.image (bien plus rapide que PNG pour courbes) ----
    buf = BytesIO()
    fig.savefig(
        buf,
        format="jpg",
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={"quality": JPEG_QUALITY, "optimize": True}
    )
    buf.seek(0)
    ph_fig.image(buf, use_container_width=True)

    # ---- Titre Matplotlib sur 2 lignes (blanc) ----
    rmse_aic = (sse_aic / cnt_aic) ** 0.5 if cnt_aic > 0 else float("nan")
    rmse_rms = (sse_rms / cnt_rms) ** 0.5 if cnt_rms > 0 else float("nan")
    ax.set_title(
        f"Internet traffic forecasting\n"
        f"RMSE AIC={rmse_aic:.4f} · RMSE Backtesting={rmse_rms:.4f}",
        color="white",
        fontsize=18
    )

    # Cadence : « sleep » seulement si on est en avance
    elapsed = time.time() - now
    if elapsed < REDRAW_EVERY:
        time.sleep(min(REDRAW_EVERY - elapsed, 0.002))