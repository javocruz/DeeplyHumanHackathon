import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
METADATA_PATH = "column_metadata.csv"
MAX_COLS = 5
TOP_N_FOR_RANDOM = 20

# =========================
# CARGAR METADATA
# =========================
meta_df = pd.read_csv(METADATA_PATH)

# =========================
# FUNCIÓN DE SCORING BASADA EN PRIORIDADES NUMÉRICAS
# =========================
def compute_score_from_priorities(meta_df, user_scores):
    """
    user_scores: dict con dimensiones y su importancia
    Ej: {"cultural":10, "economic":10, "mental":10, "physical":8.8, "social":10, "environmental":10}
    """
    df = meta_df.copy()
    df["dynamic_score"] = 0.0
    
    # Normalizar las puntuaciones del usuario a 0-1
    max_score = max(user_scores.values())
    user_norm = {k: v/max_score for k, v in user_scores.items()}
    
    for idx, row in df.iterrows():
        score = 0.0
        # 1️⃣ Coincidencia con primary_label
        primary = row["primary_label"].lower()
        if primary in user_norm:
            score += row["confidence"] * user_norm[primary]
        
        # 2️⃣ Coincidencia con secondary_labels
        for sec in row["secondary_labels"]:
            sec = sec.lower()
            if sec in user_norm:
                score += 0.5 * row["confidence"] * user_norm[sec]  # peso menor para secundarias
        
        df.at[idx, "dynamic_score"] = score

    # Orden descendente
    df = df.sort_values("dynamic_score", ascending=False)
    return df

# =========================
# FUNCIÓN DE SELECCIÓN CON ALEATORIEDAD
# =========================
def select_columns_with_randomness(df, max_cols=MAX_COLS, top_n=TOP_N_FOR_RANDOM):
    df_top = df.head(top_n).copy() if len(df) > top_n else df.copy()
    scores = df_top["dynamic_score"].values
    probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores))/len(scores)
    
    n_select = min(max_cols, len(df_top))
    selected_indices = np.random.choice(len(df_top), size=n_select, replace=False, p=probs)
    return df_top.iloc[selected_indices]["column"].tolist()

# =========================
# EJEMPLO DE USUARIO
# =========================
user_scores = {
    "cultural": 10,
    "economic": 10,
    "environmental": 10,
    "mental health": 10,
    "physical health": 8.8,
    "social wellbeing": 10
}

scored_df = compute_score_from_priorities(meta_df, user_scores)
selected_columns = select_columns_with_randomness(scored_df)

print("Recommended columns based on user priorities:")
for col in selected_columns:
    print(" -", col)
