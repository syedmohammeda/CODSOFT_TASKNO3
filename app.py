"""Streamlit interface for the saved Iris classifier."""

from pathlib import Path
import base64
import math

try:
    import joblib
    import pandas as pd
    import streamlit as st
except ImportError as error:
    raise SystemExit(
        "A required package is missing. Activate your virtual environment and run "
        "'python -m pip install -r requirements.txt'."
    ) from error


MODEL_PATH = Path(__file__).resolve().parent / "models" / "iris_model.pkl"
BACKGROUND_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "iris_background.jpg"

st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸")

try:
    image_data = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("ascii")
except OSError as error:
    st.error(f"Background image could not be loaded: {error}")
    st.stop()

st.markdown(
    f"""
    <style>
    :root {{
        --iris-ink: #f8f5e9;
        --iris-muted: #d1ddd2;
        --iris-gold: #e7cb7a;
        --iris-deep: #102821;
    }}
    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(90deg, rgba(8, 27, 23, 0.94) 0%, rgba(8, 27, 23, 0.78) 48%, rgba(8, 27, 23, 0.34) 100%),
            url("data:image/jpeg;base64,{image_data}") center 38% / cover fixed no-repeat;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{
        max-width: 1180px;
        padding-top: 4rem;
        padding-bottom: 2.5rem;
    }}
    h1 {{
        color: var(--iris-ink) !important;
        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 3rem !important;
        font-weight: 500 !important;
        line-height: 1.08 !important;
    }}
    [data-testid="stAppViewContainer"] h3 {{ color: var(--iris-ink) !important; }}
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label {{ color: var(--iris-muted); }}
    [data-testid="stCaptionContainer"] p {{ color: #d8c98d !important; }}
    [data-testid="stForm"] {{
        padding: 1.35rem 1.5rem 1.5rem;
        border: 1px solid rgba(237, 230, 198, 0.26);
        border-radius: 10px;
        background: rgba(10, 31, 26, 0.78);
        backdrop-filter: blur(14px);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
    }}
    [data-testid="stNumberInput"] input {{
        color: #17362e !important;
        background: rgba(250, 248, 239, 0.97) !important;
        border-color: rgba(231, 203, 122, 0.6) !important;
        border-radius: 6px !important;
    }}
    [data-testid="stFormSubmitButton"] button {{
        min-height: 2.8rem;
        color: #162b25;
        background: var(--iris-gold);
        border: 0;
        border-radius: 6px;
        font-weight: 700;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{
        color: #102821;
        background: #f1dda0;
        border: 0;
    }}
    [data-testid="stProgress"] > div > div {{ background-color: var(--iris-gold); }}
    @media (max-width: 760px) {{
        .block-container {{ padding: 1.5rem 1rem 2rem; }}
        h1 {{ font-size: 2.35rem !important; }}
        div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 1rem; }}
        div[data-testid="stColumn"] {{ min-width: min(100%, 280px); }}
        [data-testid="stForm"] {{ padding: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("Model file not found. Run `python src/train_model.py` in the project terminal first.")
    st.stop()

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
except (OSError, KeyError, ValueError) as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()

intro, controls = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")

with intro:
    st.caption("IRIS FLOWER  /  SPECIES IDENTIFICATION")
    st.title("Iris Flower Classification")
    st.write("Enter the flower measurements to predict its species.")
    st.caption(
        "Photo: Danielle Langlois, Wikimedia Commons, "
        "[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)."
    )

with controls:
    with st.form("measurement_form"):
        st.subheader("Flower measurements")
        first_row = st.columns(2)
        second_row = st.columns(2)
        with first_row[0]:
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.9, step=0.1)
        with first_row[1]:
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
        with second_row[0]:
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=5.1, step=0.1)
        with second_row[1]:
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=1.8, step=0.1)
        predict_clicked = st.form_submit_button("Predict Species", type="primary", use_container_width=True)

    if predict_clicked:
        measurements = [sepal_length, sepal_width, petal_length, petal_width]
        if any(not math.isfinite(value) or value <= 0 for value in measurements):
            st.error("Enter a finite measurement greater than zero in every field.")
        else:
            try:
                input_data = pd.DataFrame([measurements], columns=model_bundle["feature_names"])
                predicted_species = model.predict(input_data)[0]
                st.success(f"Predicted Species: {predicted_species}")
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    st.caption("Prediction probabilities")
                    for species, probability in zip(model.classes_, probabilities):
                        st.progress(
                            float(probability),
                            text=f"{species.removeprefix('Iris-')}  {probability:.1%}",
                        )
            except (ValueError, OSError) as error:
                st.error(f"Prediction failed: {error}")