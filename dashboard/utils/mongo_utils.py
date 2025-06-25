# File: utils/mongo_utils.py
# ----------------------------
import pandas as pd
import streamlit as st

def fetch_wamo_df(lake_name, mapping, collection):
    wamo_id = mapping.get(lake_name)
    if not wamo_id:
        st.warning("No WAMO ID mapped for selected lake.")
        return pd.DataFrame()
    wamo_docs = list(collection.find({"wamo_id": wamo_id}))
    return pd.DataFrame(wamo_docs)

def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )
    return df

def generate_comparison_prompt(df_manual, df_wamo):
    return f"""
You are a scientific data analyst. Compare manually collected water quality data (CSV) with WAMO sensor data.

Generate a table like:
| Parameter | Manual Avg | WAMO Avg | Difference | Trend |

Manual Sample (first 5 rows):
{df_manual.head(5).to_string(index=False)}

WAMO Sensor Data (first 5 rows):
{df_wamo.head(5).to_string(index=False)}
"""
