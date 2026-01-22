# =========================
# prototype_complete.py
# =========================

import pandas as pd
import os
import random

# =========================
# CONFIG
# =========================
CSV_PATH = "merged_output.csv"
METADATA_PATH = "column_metadata.csv"

BUILD_METADATA = True  # ⚠️ poner False después de la primera corrida

DIMENSIONS = [
    "mental health",
    "physical health",
    "economy",
    "education",
    "environment",
    "social wellbeing",
    "safety",
    "demographics"
]

MAX_COLS = 5  # número máximo de columnas a recomendar

# =========================
# 1️⃣ LOAD DATA
# =========================
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
metric_columns = [c for c in df.columns if c not in ["geo", "time"]]
print(f"Found {len(metric_columns)} metric columns")

# =========================
# 2️⃣ UTILS
# =========================
def humanize(col):
    return (
        col.replace("_", " ")
           .replace("percent", "%")
           .replace("per", "per")
           .replace("age", "age")
           .strip()
    )

# =========================
# 3️⃣ BUILD METADATA (RUN ONCE)
# =========================
if BUILD_METADATA:
    from transformers import pipeline

    print("Building column metadata (this runs once)...")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    metadata = []

    for col in metric_columns:
        text = humanize(col)

        result = classifier(
            text,
            candidate_labels=DIMENSIONS,
            multi_label=False
        )

        metadata.append({
            "column": col,
            "primary_label": result["labels"][0],
            "secondary_labels": [lbl for lbl in result["labels"][1:3]],  # 2 etiquetas secundarias
            "confidence": float(result["scores"][0]),
            "all_confidences": result["scores"][:3],
            "description": text,
            "unit": "percent" if "percent" in col else "count",
            "tags": col.lower().split("_")  # para inferencia de intereses
        })

        print(f"✓ {col} → {result['labels'][0]} (confidence: {result['scores'][0]:.2f})")

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(METADATA_PATH, index=False)
    print("Metadata saved to", METADATA_PATH)