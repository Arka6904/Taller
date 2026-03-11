import json, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Credit Score (ANN + PCA)", page_icon="💳", layout="centered")

MODEL_PATH = Path("model.joblib")
META_PATH  = Path("meta.json")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Faltan 'model.joblib' y/o 'meta.json'.")
    st.stop()

pipe = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

num_cols = meta["numeric_cols"]
cat_cols = meta["categorical_cols"]
ranges  = meta["ranges"]
choices = meta["choices"]

st.title("Clasificación de Credit Score")
st.caption("Pipeline: imputación, escalado, PCA y MLP dentro del modelo.")

st.sidebar.header("Entrada (escala original)")
inputs = {}

# NUMÉRICOS
for c in num_cols:
    r = ranges.get(c)
    if r is not None:
        mn, mx = r["min"], r["max"]
        step = 1.0 if (mx - mn) > 100 else 0.01
        default = float((mn + mx) / 2)
        inputs[c] = st.sidebar.slider(c, float(mn), float(mx), default, step)
    else:
        inputs[c] = st.sidebar.number_input(c, value=0.0)

# CATEGÓRICOS
for c in cat_cols:
    opts = choices.get(c, [])
    sel = st.sidebar.selectbox(c, ["<Otro>"] + opts, index=1 if opts else 0)
    if sel == "<Otro>":
        sel = st.sidebar.text_input(f"{c} (Otro)", value="")
    inputs[c] = sel

df_user = pd.DataFrame([inputs])

if st.sidebar.button("Predecir"):
    pred = pipe.predict(df_user)[0]
    st.subheader("Resultado")
    st.metric("Credit Score predicho", int(pred))
