import pandas as pd
import numpy as np
import ast
import os
import logging
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Windows safety
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------
# Safe parsing helpers
# ---------------------------------------------------------------------
def safe_parse(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, dict)):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    val_str = str(val).strip()
    if val_str.startswith("[") or val_str.startswith("{"):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, (list, dict)):
                return parsed
        except Exception:
            pass
    return val_str

def flatten_column(val, key=None):
    if isinstance(val, list):
        flattened = []
        for item in val:
            if isinstance(item, dict) and 'name' in item:
                flattened.append(item['name'])
            elif isinstance(item, str):
                flattened.append(item)
        return flattened
    elif isinstance(val, dict):
        if key and key in val:
            return val[key]
        return str(val)
    else:
        return val

# ---------------------------------------------------------------------
# Column cleaning
# ---------------------------------------------------------------------
def clean_column(df_col):
    col_name, col_data = df_col
    if pd.api.types.is_numeric_dtype(col_data):
        col_data = pd.to_numeric(col_data, errors='coerce').fillna(0)
        if pd.api.types.is_integer_dtype(col_data):
            col_data = col_data.astype(int)
        return col_name, col_data, None

    elif pd.api.types.is_object_dtype(col_data):
        col_data_parsed = col_data.apply(safe_parse)
        col_flat = None
        if col_data_parsed.apply(lambda x: isinstance(x, (list, dict))).any():
            col_flat = col_data_parsed.apply(flatten_column)
        return col_name, col_data_parsed, col_flat

    else:
        col_data = col_data.fillna(0)
        return col_name, col_data, None

# ---------------------------------------------------------------------
# Main cleaner with progress & timing
# ---------------------------------------------------------------------
def clean_goodreads_df_parallel(df: pd.DataFrame, save=True, out_dir="../data/processed", file_name="goodreads_cleaned.parquet") -> pd.DataFrame:
    logging.info(f"🔹 Cleaning Goodreads data (rows={len(df)}, cols={len(df.columns)})")
    df = df.copy()

    columns = [(col, df[col]) for col in df.columns]
    cleaned = []

    try:
        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(clean_column, columns), total=len(columns), desc="🧹 Cleaning columns"):
                cleaned.append(result)
    except Exception as e:
        logging.warning(f"⚠️ Parallel cleaning failed: {e}. Falling back to sequential mode.")
        cleaned = [clean_column(c) for c in tqdm(columns, desc="🧹 Sequential cleaning")]

    for col_name, col_data, col_flat in cleaned:
        df[col_name] = col_data
        if col_flat is not None:
            df[f"{col_name}_flat"] = col_flat.apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x))

    if save:
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, file_name)
        df.to_parquet(file_path, engine="pyarrow", index=False)
        logging.info(f"✅ Cleaned data saved to {file_path}")

        # Save schema snapshot
        schema_path = os.path.join(out_dir, f"{file_name}_schema.json")
        schema_info = {col: str(df[col].dtype) for col in df.columns}
        pd.Series(schema_info).to_json(schema_path, indent=2)
        logging.info(f"📄 Schema saved to {schema_path}")

    return df