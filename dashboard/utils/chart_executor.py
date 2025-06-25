import streamlit as st
import plotly.express as px

def get_fig_from_code(code, df):
    local_vars = {"df": df.copy(), "px": px}
    try:
        exec(code, {}, local_vars)
        fig = local_vars.get("fig")
        if not fig:
            raise ValueError("⚠️ No chart generated. Ensure code assigns to `fig`.")
        return fig
    except Exception as e:
        st.error(f"⚠️ Error generating chart: {e}")
        if st.button("🔁 Retry Generating Chart", key=f"retry_button_{hash(code)}"):
            st.rerun()
        return None
