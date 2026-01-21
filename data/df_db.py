import pandas as pd
from supabase import create_client
import math
import time
import numpy as np

SUPABASE_URL = "https://ziafykczhnjcfcmxleba.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppYWZ5a2N6aG5qY2ZjbXhsZWJhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTAwNjAwNywiZXhwIjoyMDg0NTgyMDA3fQ.pdoV-n6DRJFk9WVHRGSBntBWy-CVRSRCJHGMnhLnscs"

TABLE_NAME = "Countries datapoints"
CSV_PATH = "merged_output.csv"

BATCH_SIZE = 500
SLEEP_BETWEEN = 0.2

START_BATCH = 146


def clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    # Asegura que geo/time existen
    df = df.dropna(subset=["geo", "time"]).copy()

    # Normaliza tipos base
    df["geo"] = df["geo"].astype(str).str.strip()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["time"] = df["time"].astype(int)

    # Convierte inf/-inf a NaN, luego NaN -> None
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.astype(object).where(pd.notnull(df), None)

    return df


def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    df = pd.read_csv(CSV_PATH, low_memory=False)

    df = clean_for_json(df)

    rows = df.to_dict(orient="records")

    total = len(rows)
    total_batches = math.ceil(total / BATCH_SIZE)

    print(f"Filas a subir: {total}")
    print(f"Batches: {total_batches} (size={BATCH_SIZE})")

    for i in range(START_BATCH-1, total_batches):
        batch = rows[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        for r in batch:
            for k, v in list(r.items()):
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    r[k] = None

        supabase.table(TABLE_NAME).upsert(batch, on_conflict="geo,time").execute()

        print(f"✅ Batch {i+1}/{total_batches} subido ({len(batch)} filas)")
        time.sleep(SLEEP_BETWEEN)

    print("🚀 Carga completada")


if __name__ == "__main__":
    main()
