# =========================
# forms_recommendation.py
# =========================

import os
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
METADATA_PATH = "column_metadata.csv"  # CSV generado por prototype_complete.py
MAX_COLS = 5                            # máximo de columnas a recomendar
DIMENSIONS = [
    "mental health", "physical health", "economy", "education",
    "environment", "social wellbeing", "safety", "demographics"
]
TOP_N_FOR_RANDOM = 20  # solo aplicar aleatoriedad dentro de top N por score

# =========================
# CARGAR METADATA
# =========================
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}. Ejecuta primero prototype_complete.py")

meta_df = pd.read_csv(METADATA_PATH)

# =========================
# FUNCIÓN DE SCORING DINÁMICO
# =========================
def compute_dynamic_scores(meta_df, user_form, weights=None, text_inference_weight=1.2):
    """
    Calcula un score dinámico por columna basado en:
    - Prioridades explícitas e implícitas
    - País
    - Tiempo
    - Confidence del modelo
    """
    if weights is None:
        weights = {
            "priority": 1.0,
            "country": 0.2,
            "time": 0.1
        }

    df = meta_df.copy()
    df["dynamic_score"] = 0.0

    # -------------------
    # 1️⃣ Prioridades explícitas
    # -------------------
    for priority in user_form.get("priorities", []):
        mask = df["primary_label"] == priority
        df.loc[mask, "dynamic_score"] += df.loc[mask, "confidence"] * weights["priority"]

    # -------------------
    # 2️⃣ Inferencia desde texto libre
    # -------------------
    text_input = user_form.get("text_input", "").lower()
    if text_input:
        for idx, row in df.iterrows():
            match_score = 0.0
            # Coincidencia con tags
            for tag in row["tags"]:
                if tag in text_input:
                    match_score += 1.0
            # Coincidencia con secondary_labels
            for sec_label in row["secondary_labels"]:
                if sec_label.lower() in text_input:
                    match_score += 0.8
            if match_score > 0:
                df.at[idx, "dynamic_score"] += row["confidence"] * text_inference_weight * match_score

    # -------------------
    # 3️⃣ País
    # -------------------
    country = user_form.get("country")
    if "available_countries" in df.columns and country:
        df["country_score"] = df["available_countries"].apply(lambda x: 1.0 if country in x else 0.0)
        df["dynamic_score"] += df["country_score"] * weights["country"]

    # -------------------
    # 4️⃣ Tiempo
    # -------------------
    time_focus = user_form.get("time_focus")
    if "time_range" in df.columns and time_focus:
        def time_score(ranges):
            for r in ranges:
                start, end = map(int, r.split("-"))
                if time_focus == "future" and end > 2030:
                    return 1.0
                elif time_focus == "past" and end <= 2030:
                    return 1.0
            return 0.0
        df["time_score"] = df["time_range"].apply(time_score)
        df["dynamic_score"] += df["time_score"] * weights["time"]

    # -------------------
    # Ordenar por score descendente
    # -------------------
    df = df.sort_values("dynamic_score", ascending=False)
    return df

# =========================
# FUNCIÓN DE SELECCIÓN CON ALEATORIEDAD CONTROLADA
# =========================
def select_dynamic_columns(df, max_cols=MAX_COLS, top_n=TOP_N_FOR_RANDOM):
    """
    Selecciona columnas recomendadas:
    - max_cols: número máximo de columnas
    - solo aleatoriedad dentro de las top N columnas por score
    """
    df_top = df.head(top_n).copy() if len(df) > top_n else df.copy()
    
    scores = df_top["dynamic_score"].values
    if scores.sum() == 0:
        probs = np.ones(len(scores)) / len(scores)
    else:
        probs = scores / scores.sum()

    n_select = min(max_cols, len(df_top))
    selected_indices = np.random.choice(len(df_top), size=n_select, replace=False, p=probs)
    
    return df_top.iloc[selected_indices]["column"].tolist()

# =========================
# EJEMPLO DE USUARIO
# =========================
user_form = {
    "country": "MEX",
    "priorities": ["economy"],       # Prioridades explícitas
    "text_input": "work and jobs",   # Texto libre para inferir intereses
    "time_focus": "past"             # past/future
}

# =========================
# CÁLCULO DINÁMICO Y SELECCIÓN
# =========================
scored_df = compute_dynamic_scores(meta_df, user_form)
selected_columns = select_dynamic_columns(scored_df)

print("Recommended columns (dynamic + controlled randomness + inferred priorities):")
for col in selected_columns:
    print(" -", col)
