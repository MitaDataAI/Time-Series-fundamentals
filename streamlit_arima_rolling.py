
import io
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Rolling — Simulation temps réel", layout="wide")

st.title("🎛️ Simulation Rolling ARIMA en *vrai faux* temps réel")
st.markdown(
    "Cette app simule le **déroulé en streaming** d'une série test et calcule à chaque pas une "
    "prévision *1-step ahead* avec un **modèle ARIMA déjà entraîné** (ou entraîné à la volée).\n"
    "👉 Chargez `y_train` et `y_test` (CSV avec colonnes `date`, `value`).\n"
    "👉 Optionnel : chargez un modèle `res` picklé (statsmodels ARIMA/SARIMAX Results)."
)

with st.expander("📥 Importer les données (CSV)"):
    c1, c2 = st.columns(2)
    with c1:
        y_train_file = st.file_uploader("y_train.csv", type=["csv"], key="ytrain")
        y_test_file = st.file_uploader("y_test.csv", type=["csv"], key="ytest")
    with c2:
        model_file = st.file_uploader("Modèle ARIMA picklé (.pkl)", type=["pkl","pickle"], key="model")
        date_fmt = st.text_input("Format de date (optionnel, ex: %Y-%m-%d)", value="")

def read_series(f):
    if f is None:
        return None
    df = pd.read_csv(f)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "value" not in cols:
        st.error(f"Le fichier '{getattr(f, 'name', '??')}' doit contenir les colonnes 'date' et 'value'.")
        return None
    if date_fmt:
        idx = pd.to_datetime(df[cols["date"]], format=date_fmt, errors="coerce")
    else:
        idx = pd.to_datetime(df[cols["date"]], errors="coerce")
    s = pd.Series(df[cols["value"]].values, index=idx).sort_index()
    s = s[~s.index.isna()]
    return s

y_train = read_series(y_train_file)
y_test = read_series(y_test_file)

# Démo synthétique si rien n'est fourni
if y_train is None or y_test is None:
    idx_train = pd.date_range("2023-01-01", periods=200, freq="D")
    idx_test = pd.date_range(idx_train[-1] + pd.Timedelta(days=1), periods=90, freq="D")
    rng = np.random.default_rng(0)
    y_train = pd.Series(np.cumsum(rng.normal(0, 1, len(idx_train))), index=idx_train, name="y_train")
    base = y_train.iloc[-1]
    drift = np.linspace(0, 7, len(idx_test))
    noise = rng.normal(0, 1.2, len(idx_test))
    y_test = pd.Series(base + drift + noise, index=idx_test, name="y_test")
    st.info("Aucune donnée fournie — utilisation d'un exemple synthétique.")

# --- Choix du modèle ---
st.sidebar.header("Paramètres du modèle")
model_src = st.sidebar.radio("Source du modèle", ["Upload .pkl", "Entraîner sur y_train"], index=0)

order = None
if model_src == "Entraîner sur y_train":
    p = st.sidebar.number_input("p", min_value=0, value=1, step=1)
    d = st.sidebar.number_input("d", min_value=0, value=0, step=1)
    q = st.sidebar.number_input("q", min_value=0, value=1, step=1)
    order = (p, d, q)

# Charger / entraîner le modèle initial 'res' (sur y_train)
res = None
if model_src == "Upload .pkl" and model_file is not None:
    try:
        res = pickle.load(model_file)
    except Exception as e:
        st.error(f"Impossible de charger le modèle picklé: {e}")
if res is None:
    # Entraîner simple ARIMA si pas d'upload
    try:
        res = ARIMA(y_train, order=order or (1,0,1)).fit()
    except Exception as e:
        st.error(f"Erreur d'entraînement ARIMA ({order or (1,0,1)}): {e}")
        st.stop()

# --- Contrôles de simulation ---
st.sidebar.header("Simulation temps réel")
max_steps = len(y_test)
if "step" not in st.session_state:
    st.session_state.step = 1
if "auto" not in st.session_state:
    st.session_state.auto = False

c0, c1, c2, c3, c4 = st.sidebar.columns([1,1,1,1,1])
if c0.button("⏮️"):
    st.session_state.step = 1
if c1.button("⏪ -10"):
    st.session_state.step = max(1, st.session_state.step - 10)
if c2.button("⏭️ +1"):
    st.session_state.step = min(max_steps, st.session_state.step + 1)
if c3.button("⏭️ +10"):
    st.session_state.step = min(max_steps, st.session_state.step + 10)
if c4.button("⏭⏭ Fin"):
    st.session_state.step = max_steps

st.session_state.auto = st.sidebar.toggle("▶️ Lecture auto", value=st.session_state.auto, key="auto_toggle")
speed = st.sidebar.slider("Vitesse (ms/pas)", min_value=100, max_value=2000, value=600, step=50)

# Auto-refresh (si activé)
try:
    token = st.autorefresh(interval=int(speed), limit=None, key="autorefresh_token")
    if st.session_state.auto and st.session_state.step < max_steps:
        st.session_state.step += 1
except Exception:
    pass  # st.autorefresh peut ne pas être dispo selon la version

steps = st.session_state.step
st.markdown(f"**Pas actuel :** {steps} / {max_steps}")

# --- Rolling Forecast (1-step ahead) jusqu'au pas 'steps' ---
# On clone l'objet 'res' pour éviter d'accumuler les appends à chaque exécution.
# Simplification: on relance depuis le modèle sur y_train et on déroule 'steps' fois.
res_rolling = res
preds, low, high = [], [], []
idxs = list(y_test.index[:steps])
for ts, y_true in y_test.iloc[:steps].items():
    fc = res_rolling.get_forecast(steps=1)
    yhat = float(fc.predicted_mean.iloc[0])
    ci = fc.conf_int(alpha=0.05).iloc[0]
    preds.append(yhat)
    low.append(float(ci.iloc[0]))
    high.append(float(ci.iloc[1]))
    # Append la vraie obs au modèle sans refit
    new_obs = pd.Series([y_true], index=[ts])
    try:
        res_rolling = res_rolling.append(new_obs, refit=False)
    except Exception as e:
        st.error(f"Erreur pendant append() : {e}")
        st.stop()

rolling_preds = pd.Series(preds, index=y_test.index[:steps])
conf_low = pd.Series(low, index=y_test.index[:steps])
conf_high = pd.Series(high, index=y_test.index[:steps])

# RMSE sur la fenêtre révélée
from sklearn.metrics import mean_squared_error
rmse_rolling = float(np.sqrt(mean_squared_error(y_test.iloc[:steps], rolling_preds)))

# --- Visualisation ---
fig = plt.figure(figsize=(14, 8))
plt.plot(y_train.index, y_train.values, linewidth=1.5, label="TRAIN")
plt.plot(y_test.index[:steps], y_test.values[:steps], linewidth=1.8, label="TEST (révélé)")
plt.plot(rolling_preds.index, rolling_preds.values, linewidth=2, linestyle="--", label="Prévision ARIMA rolling")
plt.fill_between(conf_low.index, conf_low.values, conf_high.values, alpha=0.3, label="IC 95%")
plt.axvline(y_test.index[0], linestyle="--", linewidth=1, alpha=0.8)
plt.title("Backtesting et Prévisions ARIMA — Simulation Rolling")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.legend(loc="upper left")
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# --- KPIs ---
k1, k2, k3 = st.columns(3)
k1.metric("RMSE (fenêtre révélée)", f"{rmse_rolling:.4f}")
k2.metric("Obs révélées", f"{steps}")
try:
    k3.metric("Dernière prévision", f"{rolling_preds.iloc[-1]:.3f}", help="yhat au dernier pas")
except Exception:
    k3.metric("Dernière prévision", "—")

with st.expander("🔎 Détails dernière itération"):
    try:
        st.write("**Timestamp courant**:", str(y_test.index[steps-1]))
        st.write("**y_true**:", float(y_test.iloc[steps-1]))
        st.write("**yhat**:", float(rolling_preds.iloc[-1]))
        st.write("**IC 95%**:", (float(conf_low.iloc[-1]), float(conf_high.iloc[-1])))
    except Exception:
        st.write("—")

# --- Téléchargements ---
def to_csv_bytes(s):
    buf = io.StringIO()
    s.to_csv(buf, header=["value"])
    return buf.getvalue().encode("utf-8")

cdl, cdr = st.columns(2)
cdl.download_button("📥 Exporter prédictions (CSV)", data=to_csv_bytes(rolling_preds), file_name="rolling_preds.csv", mime="text/csv")
cdr.download_button("📥 Exporter intervalles (CSV)", data=to_csv_bytes(pd.DataFrame({"conf_low": conf_low, "conf_high": conf_high})["conf_low"]), file_name="conf_low.csv", mime="text/csv")
