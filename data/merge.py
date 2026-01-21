import os
import glob
import pandas as pd


def load_and_merge_folder(folder_path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en: {folder_path}")

    dfs = []
    used_metric_names = set()

    for fp in files:
        df = pd.read_csv(fp)

        # Validación mínima
        if "geo" not in df.columns or "time" not in df.columns:
            raise ValueError(f"El archivo {os.path.basename(fp)} no tiene columnas 'geo' y 'time'.")

        # Quitar filas inválidas (geo o time vacíos) -> NO se agregan
        df = df.dropna(subset=["geo", "time"])
        if df.empty:
            continue

        # Tipos consistentes
        df["geo"] = df["geo"].astype(str).str.strip()
        df["time"] = pd.to_numeric(df["time"], errors="coerce")

        # Si time no se pudo convertir, se elimina
        df = df.dropna(subset=["time"])
        if df.empty:
            continue
        df["time"] = df["time"].astype("Int64")

        # Detectar columna de métrica (la 3ra columna, aparte de geo/time/gender si existe)
        metric_cols = [c for c in df.columns if c not in ("geo", "time", "gender")]
        if len(metric_cols) != 1:
            raise ValueError(
                f"Esperaba exactamente 1 columna de métrica además de geo/time/(gender), "
                f"pero {os.path.basename(fp)} tiene: {metric_cols}"
            )
        value_col = metric_cols[0]

        # (Opcional) si quieres ignorar filas donde el valor está vacío, descomenta:
        # df = df.dropna(subset=[value_col])
        # if df.empty:
        #     continue

        # Si existe gender (0/1), sumar hombres+mujeres por geo+time
        if "gender" in df.columns:
            df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

            df = (
                df.groupby(["geo", "time"], as_index=False)[value_col]
                  .sum(min_count=1)
            )
        else:
            # Asegura que el valor sea numérico si aplica (si no, queda como object)
            df[value_col] = pd.to_numeric(df[value_col], errors="ignore")

        # Si por cualquier razón quedan duplicados por geo+time, colapsarlos (último no-null)
        if df.duplicated(subset=["geo", "time"]).any():
            df = (
                df.sort_values(["geo", "time"])
                  .groupby(["geo", "time"], as_index=False)[value_col]
                  .last()
            )

        # Evitar colisiones de nombre de columna (si dos archivos traen mismo nombre)
        base = os.path.splitext(os.path.basename(fp))[0]
        out_col = value_col
        if out_col in used_metric_names:
            out_col = f"{value_col}__{base}"
        used_metric_names.add(out_col)

        df = df.rename(columns={value_col: out_col})
        dfs.append(df)

    if not dfs:
        raise ValueError("Todos los CSV quedaron vacíos después de limpiar geo/time.")

    # Merge sucesivo por geo+time
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["geo", "time"], how="outer")

    merged = merged.sort_values(["geo", "time"]).reset_index(drop=True)

    # Garantía: 1 fila por geo+time
    if merged.duplicated(subset=["geo", "time"]).any():
        raise RuntimeError("El resultado final tiene duplicados por geo+time (no debería pasar).")

    return merged


if __name__ == "__main__":
    folder = "/Users/josemiguelreyesalegria/Desktop/ddf--gapminder--systema_globalis-master/countries-etc-datapoints"
    result = load_and_merge_folder(folder)
    print(result.head())
    result.to_csv("merged_output.csv", index=False)
