import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage

def render(df):
    st.subheader("🔍 Preview Uploaded Data")

    # 🧠 AI LLM Summary
    with st.expander("🧠 AI Summary of Your Dataset", expanded=True):
        try:
            df_preview = df.head(5).to_string(index=False)
            missing_summary = df.isnull().mean().round(3) * 100
            missing_str = missing_summary.to_string()

            numeric_df = df.select_dtypes(include='number')
            stats = numeric_df.describe().T[['mean', 'std', 'min', 'max']].to_string()

            summary_prompt = f"""
You are a data quality assistant for environmental datasets.

The user has uploaded a dataset with shape {df.shape[0]} rows × {df.shape[1]} columns.

Here are the first few rows:
{df_preview}

Here is the % of missing values per column:
{missing_str}

Here are basic statistics for numeric columns:
{stats}

Write a short summary in bullet points highlighting:
- What this dataset looks like (parameters, types)
- Any data quality concerns (nulls, spikes, low DO, etc.)
- Anything noteworthy
"""
            response = st.session_state["model"].invoke([HumanMessage(content=summary_prompt)])
            st.markdown(response.content)
        except Exception as e:
            st.warning(f"⚠️ Could not generate LLM summary: {e}")

    # 📋 First 100 rows
    with st.expander("🔍 View First 100 Rows"):
        st.dataframe(df.head(100))
        st.markdown(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")

    # 🔠 Column types
    with st.expander("🔠 Column Data Types"):
        st.dataframe(
            df.dtypes.astype(str)
            .reset_index()
            .rename(columns={"index": "Column", 0: "Data Type"})
        )

    # ❓ Missing value summary
    with st.expander("❓ Missing Value Summary"):
        st.dataframe(
            df.isnull().sum()
            .reset_index()
            .rename(columns={"index": "Column", 0: "Missing Count"})
        )

    # 📊 Numeric summary statistics
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        with st.expander("📊 Numeric Summary Statistics"):
            stats = numeric_df.describe().T
            missing_percent = df[numeric_df.columns].isnull().mean() * 100

            summary = stats[['mean', 'std', 'min', 'max']].copy()
            summary['% Missing'] = missing_percent
            summary = summary.reset_index().rename(columns={'index': 'Column'})

            st.dataframe(summary)
    else:
        st.info("No numeric columns found for summary statistics.")

    # 🚨 Outlier detection (time series)
    if not numeric_df.empty:
        with st.expander("🚨 Outlier Detection Over Time"):
            date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            if date_cols:
                date_col = st.selectbox("Select a date/time column", date_cols)
                param_col = st.selectbox("Select a numeric column to visualize", numeric_df.columns)

                df_copy = df[[date_col, param_col]].dropna()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_copy = df_copy.dropna()

                Q1 = df_copy[param_col].quantile(0.25)
                Q3 = df_copy[param_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_copy["is_outlier"] = (
                    (df_copy[param_col] < lower_bound) | (df_copy[param_col] > upper_bound)
                )

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_copy[date_col],
                    y=df_copy[param_col],
                    mode='lines+markers',
                    name="Normal Values",
                    marker=dict(color="gray"),
                    line=dict(color="lightgray")
                ))

                fig.add_trace(go.Scatter(
                    x=df_copy[df_copy["is_outlier"]][date_col],
                    y=df_copy[df_copy["is_outlier"]][param_col],
                    mode='markers',
                    name="Possible Outliers",
                    marker=dict(color="red", size=10, symbol="circle-open")
                ))

                fig.update_layout(
                    title=f"{param_col} Over Time (Outliers Highlighted)",
                    xaxis_title=date_col,
                    yaxis_title=param_col,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No date/time column found. Add one to enable this chart.")
