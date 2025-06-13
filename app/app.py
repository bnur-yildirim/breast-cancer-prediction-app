import os
import pathlib
import joblib
import json
from typing import Dict, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras.models import load_model


@st.cache_resource(show_spinner=False)
def load_all_models(model_dir: str = "models") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    results_path = pathlib.Path(model_dir) / "model_results.csv"
    df_results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()

    models: Dict[str, Any] = {}
    for file in pathlib.Path(model_dir).glob("*.*"):
        if file.suffix == ".joblib":
            models[file.stem.replace("_", " ")] = joblib.load(file)
        elif file.suffix == ".h5":
            models[file.stem] = load_model(file, compile=False)
    return df_results, models


st.set_page_config(page_title="Breast Cancer Wisconsin Demo", layout="wide")

st.sidebar.title("Breast Cancer Wisconsin")
page = st.sidebar.radio(
    "Navigate", ["Data Visualisation", "Model Comparison", "Prediction"]
)

RESULTS_DF, ALL_MODELS = load_all_models()
DEFAULT_DATA_PATH = "../data/preprocessed_data.csv"

METRIC_FMT = {
    "Accuracy": "{:.3f}",
    "Precision": "{:.3f}",
    "Recall": "{:.3f}",
    "F1": "{:.3f}",
    "ROC_AUC": "{:.3f}",
}


def display_metrics(df: pd.DataFrame):
    st.dataframe(df.style.format(METRIC_FMT).highlight_max(axis=0, color="lightgreen"))


# DATA VISUALISATION
if page == "Data Visualisation":
    st.header("Dataset Quick Look")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif pathlib.Path(DEFAULT_DATA_PATH).exists():
        data = pd.read_csv(DEFAULT_DATA_PATH)
    else:
        st.warning("No dataset available – please upload one.")
        st.stop()

    st.subheader("First 5 rows")
    st.write(data.head())

    diag_col = "diagnosis" if "diagnosis" in data.columns else None
    if diag_col:
        st.subheader("Class distribution")
        st.bar_chart(data[diag_col].value_counts())

    st.subheader("Correlation heatmap")

    # Select only numeric columns
    numeric_df = data.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=16)
    st.pyplot(fig)


# MODEL COMPARISON
if page == "Model Comparison":
    st.header("Compare Trained Models")

    if RESULTS_DF.empty:
        st.error("No results CSV found in /models. Run the training script first.")
        st.stop()

    display_metrics(RESULTS_DF)

    st.subheader("Metric Comparison Chart")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    if all(col in RESULTS_DF.columns for col in metric_cols):
        fig, ax = plt.subplots(figsize=(10, 5))
        RESULTS_DF.set_index("Model")[metric_cols].plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        st.pyplot(fig)
    else:
        st.warning("Some metric columns are missing for chart.")

    # Buttons for selecting model by metric (best and worst)
    st.subheader("Select model automatically based on metric")

    cols = st.columns(10)  # 5 metrics * 2 buttons each (best & worst)

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

    for i, metric in enumerate(metrics):
        with cols[i * 2]:
            if st.button(f"Best {metric}"):
                st.session_state.selected_model = RESULTS_DF.loc[
                    RESULTS_DF[metric].idxmax(), "Model"
                ]
        with cols[i * 2 + 1]:
            if st.button(f"Worst {metric}"):
                st.session_state.selected_model = RESULTS_DF.loc[
                    RESULTS_DF[metric].idxmin(), "Model"
                ]


# Model selection dropdown (persistent)
chosen_model_name = st.selectbox(
    "Select a model to inspect / deploy",
    RESULTS_DF["Model"].tolist(),
    index=RESULTS_DF["Model"]
    .tolist()
    .index(
        st.session_state.selected_model
        if st.session_state.selected_model in RESULTS_DF["Model"].tolist()
        else RESULTS_DF["Model"].tolist()[0]
    ),
    key="model_dropdown",
)

# Update session state to keep in sync with dropdown
st.session_state.selected_model = chosen_model_name

st.success(f"Selected model: {st.session_state.selected_model}")


if st.session_state.selected_model:
    st.success(f"Auto-selected model: {st.session_state.selected_model}")

    model_obj = ALL_MODELS.get(chosen_model_name)

    if model_obj is None:
        st.error("Model file not found. Ensure it resides in /models.")
        st.stop()

    st.success(f"Loaded {chosen_model_name}")

    if pathlib.Path(DEFAULT_DATA_PATH).exists():
        data = pd.read_csv(DEFAULT_DATA_PATH)
        X_val = data.drop("diagnosis", axis=1)
        y_val = data["diagnosis"]
        if hasattr(model_obj, "predict"):
            if hasattr(model_obj, "predict_proba"):
                y_pred = model_obj.predict(X_val)
            else:
                y_pred = (model_obj.predict(X_val).ravel() >= 0.5).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Benign", "Malignant"]
            )
            disp.plot(ax=ax_cm, cmap="Blues")
            st.pyplot(fig_cm)


# PREDICTION
if page == "Prediction":
    st.header("Predict with Your Selected Model or All Models")

    if not ALL_MODELS:
        st.error("No models found. Train models first.")
        st.stop()

    model_name = st.selectbox("Choose a model", list(ALL_MODELS.keys()))
    model = ALL_MODELS[model_name]
    st.info(f"Model {model_name} is loaded.")

    st.markdown("Upload feature CSV (preprocessed columns only)")
    pred_file = st.file_uploader("Upload CSV", type=["csv"])

    # Predefined column means
    column_means = {
        "radius_mean": 14.1273,
        "texture_mean": 19.2896,
        "perimeter_mean": 91.9690,
        "area_mean": 654.8891,
        "smoothness_mean": 0.0964,
        "compactness_mean": 0.1043,
        "concavity_mean": 0.0888,
        "concave points_mean": 0.0489,
        "symmetry_mean": 0.1812,
        "fractal_dimension_mean": 0.0628,
        "radius_se": 0.4052,
        "texture_se": 1.2169,
        "perimeter_se": 2.8661,
        "area_se": 40.3371,
        "smoothness_se": 0.0070,
        "compactness_se": 0.0255,
        "concavity_se": 0.0319,
        "concave points_se": 0.0118,
        "symmetry_se": 0.0205,
        "fractal_dimension_se": 0.0038,
        "radius_worst": 16.2692,
        "texture_worst": 25.6772,
        "perimeter_worst": 107.2612,
        "area_worst": 880.5831,
        "smoothness_worst": 0.1324,
        "compactness_worst": 0.2543,
        "concavity_worst": 0.2722,
        "concave points_worst": 0.1146,
        "symmetry_worst": 0.2901,
        "fractal_dimension_worst": 0.0839,
    }

    def fill_missing_with_means(df):
        for col, mean_val in column_means.items():
            if col in df.columns:
                df[col] = df[col].fillna(mean_val)
        return df

    data_for_prediction = None
    if pred_file is not None:
        X_new = pd.read_csv(pred_file)
        X_new = fill_missing_with_means(X_new)
        data_for_prediction = X_new
        st.write("Preview of uploaded data:")
        st.write(X_new.head())
    else:
        st.markdown("Or manually enter feature values below:")

        st.info(
            "Each input field below is pre-filled with the **average value** from the training dataset. "
            "You can change any values to make a prediction based on your own data."
        )

        user_input = {}

        # Number of columns per row
        num_cols = 5

        # Create that many Streamlit columns
        cols = st.columns(num_cols)

        # Iterate over columns and assign inputs in a grid fashion
        for i, (col_name, mean_val) in enumerate(column_means.items()):
            col = cols[i % num_cols]
            user_input[col_name] = col.number_input(
                col_name, value=mean_val, format="%.4f", key=f"input_{col_name}"
            )

        # Convert to DataFrame so prediction works
        data_for_prediction = pd.DataFrame([user_input])

    # Single model prediction button
    if st.button("Predict with selected model"):
        model = ALL_MODELS[model_name]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data_for_prediction)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            probs = model.predict(data_for_prediction).ravel()
            preds = (probs >= 0.5).astype(int)

        out_df = data_for_prediction.copy()
        out_df["Prediction"] = preds
        out_df["Probability (Malignancy)"] = probs

        st.subheader(f"Predictions from {model_name}")
        st.dataframe(out_df.head())

        csv = out_df.to_csv(index=False).encode()
        st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

        # Show summary
        st.write(f"Summary: {np.bincount(preds)} counts of [Benign=0, Malignant=1]")
        st.write(f"Mean malignancy probability: {probs.mean():.4f}")

    # Predict with all models button
    if st.button("Predict with all models and compare"):
        if data_for_prediction is None:
            st.warning("Please upload CSV or enter data manually first.")
        else:
            all_preds = {}
            all_probs = {}

            for m_name, m_obj in ALL_MODELS.items():
                if hasattr(m_obj, "predict_proba"):
                    p = m_obj.predict_proba(data_for_prediction)[:, 1]
                    preds = (p >= 0.5).astype(int)
                else:
                    p = m_obj.predict(data_for_prediction).ravel()
                    preds = (p >= 0.5).astype(int)
                all_preds[m_name] = preds
                all_probs[m_name] = p

            # Prepare summary dataframe
            summary_list = []

            for m_name in ALL_MODELS.keys():
                preds = all_preds[m_name]
                probs = all_probs[m_name]
                benign_count = np.sum(preds == 0)
                malignant_count = np.sum(preds == 1)
                mean_prob = probs.mean()
                summary_list.append(
                    {
                        "Model": m_name,
                        "Benign Count": benign_count,
                        "Malignant Count": malignant_count,
                        "Mean Malignancy Probability": mean_prob,
                    }
                )

            summary_df = pd.DataFrame(summary_list)

            st.subheader("Prediction Summary Across Models")
            st.dataframe(summary_df.set_index("Model"))

            # Plot comparison charts
            fig, ax = plt.subplots(figsize=(10, 5))
            summary_df.set_index("Model")[["Benign Count", "Malignant Count"]].plot(
                kind="bar", stacked=True, ax=ax
            )
            plt.title("Count of Predicted Classes per Model")
            plt.ylabel("Number of Samples")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            summary_df.set_index("Model")["Mean Malignancy Probability"].plot(
                kind="bar", color="orange", ax=ax2
            )
            plt.title("Mean Malignancy Probability per Model")
            plt.ylabel("Mean Probability")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
