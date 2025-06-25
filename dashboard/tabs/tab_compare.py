# ----------------------------
# File: tabs/tab_compare.py
# ----------------------------
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
from utils.mongo_utils import fetch_wamo_df, normalize_columns, generate_comparison_prompt

def render(df, selected_lake, WAMO_MAPPING, wamo_collection):
    st.subheader("📋 LLM-Based Comparison with WAMO")

    wamo_df = fetch_wamo_df(selected_lake, WAMO_MAPPING, wamo_collection)
    if not wamo_df.empty:
        # Normalize column names
        df = normalize_columns(df)
        wamo_df = normalize_columns(wamo_df)

        # LLM Summary Prompt
        compare_prompt = generate_comparison_prompt(df, wamo_df)
        response = st.session_state["model"].invoke([HumanMessage(content=compare_prompt)])
        st.markdown(response.content)

        # Detect date columns
        def find_date_column(df):
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    return col
            return None

        date_col_manual = find_date_column(df)
        date_col_wamo = find_date_column(wamo_df)

        if date_col_manual and date_col_wamo:
            try:
                df['date'] = pd.to_datetime(df[date_col_manual])
                wamo_df['date'] = pd.to_datetime(wamo_df[date_col_wamo])

                st.subheader("📈 Matched Time Series Parameters")
                common_cols = list(set(df.columns) & set(wamo_df.columns))
                numeric_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(wamo_df[col])]

                for col in numeric_cols:
                    st.markdown(f"#### 📊 {col} (Manual vs. WAMO)")
                    merged = pd.merge(
                        df[['date', col]],
                        wamo_df[['date', col]],
                        on='date',
                        how='inner',
                        suffixes=('_manual', '_wamo')
                    )

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=merged['date'],
                        y=merged[f"{col}_manual"],
                        mode='lines+markers',
                        name=f'Manual {col}'
                    ))
                    fig.add_trace(go.Scatter(
                        x=merged['date'],
                        y=merged[f"{col}_wamo"],
                        mode='lines+markers',
                        name=f'WAMO {col}'
                    ))

                    fig.update_layout(
                        title=f"{col} Over Time",
                        xaxis_title='Date',
                        yaxis_title=col,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"📅 Error processing date columns: {e}")
        else:
            st.warning("📅 Could not detect valid date/time columns in one of the datasets.")
    else:
        st.warning("⚠️ No WAMO data found for the selected lake.")
