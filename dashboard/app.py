# --- imports ---
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import plotly.graph_objects as go


# --- config ---
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# MongoDB
client = MongoClient(st.secrets["MONGO_URI"])
wamo_collection = client["Wamoproject"]["CSV"]

# WAMO Mapping
WAMO_MAPPING = {
    "Babenhäuser See": "wamo00023",
    # Add more lake mappings here
}

# --- model setup ---
if 'model' not in st.session_state:
    st.session_state['model'] = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

# --- helpers ---

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
        if st.button("🔁 Retry Generating Chart"):
            st.rerun()
        return None

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

def fetch_wamo_df(lake_name):
    wamo_id = WAMO_MAPPING.get(lake_name)
    if not wamo_id:
        st.warning("No WAMO ID mapped for selected lake.")
        return pd.DataFrame()
    wamo_docs = list(wamo_collection.find({"wamo_id": wamo_id}))
    return pd.DataFrame(wamo_docs)

# --- UI ---
st.set_page_config(page_title="📊 AI Chart Assistant", layout="wide")
st.title("📊 AI Chart Assistant")

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    selected_lake = st.selectbox("🏞️ Select Lake for WAMO Comparison", list(WAMO_MAPPING.keys()))
    compare_with_wamo = st.checkbox("🔄 Compare with WAMO", value=False)
    temperature = st.slider("🔥 Model Temperature", 0.0, 2.0, 0.7, 0.1)

# File Handling
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Preview Data", "📊 Suggested Charts", "📋 Compare with WAMO", "💬 Chat with Assistant"])

    # Preview
    with tab1:
        st.subheader("🔍 Preview Uploaded Data")
        st.dataframe(df.head(100))

    # Chart Suggestions
    with tab2:
        st.subheader("📊 AI-Suggested Charts")
        df_preview = df.head(100).to_string(index=False)
        suggestion_prompt = """
You are a data visualization expert. Based on the first 100 rows of this dataset, suggest 2–3 insightful Plotly Express visualizations.
Return valid code blocks only (no explanations), like:
```python
fig = px.line(df, x="Time", y="pH", title="pH over Time")
```
        """
        suggestion_response = st.session_state["model"].invoke([
            HumanMessage(content=suggestion_prompt + "\n\nCSV Preview:\n" + df_preview)
        ])
        code_blocks = re.findall(r"```(?:[Pp]ython)?(.*?)```", suggestion_response.content, re.DOTALL)
        for i, code in enumerate(code_blocks):
            fig = get_fig_from_code(code.strip(), df)
            if fig:
                st.subheader(f"Suggested Chart #{i + 1}")
                st.plotly_chart(fig, use_container_width=True)

    # Comparison with WAMO
    with tab3:
        st.subheader("📋 LLM-Based Comparison with WAMO")
        if compare_with_wamo:
            wamo_df = fetch_wamo_df(selected_lake)
            if not wamo_df.empty:

            # Normalize column names
                def normalize_columns(df):
                    df.columns = (
                        df.columns.str.strip()
                                .str.lower()
                                .str.replace(" ", "_")
                                .str.replace("-", "_")
                )
                    return df

            df = normalize_columns(df)
            wamo_df = normalize_columns(wamo_df)

            # LLM comparison table
            compare_prompt = generate_comparison_prompt(df, wamo_df)
            response = st.session_state["model"].invoke([HumanMessage(content=compare_prompt)])
            st.markdown(response.content)

            # Auto date column detection
            def find_date_column(df):
                for col in df.columns:
                    if "date" in col or "time" in col:
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
                    numeric_cols = [col for col in common_cols
                                    if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(wamo_df[col])]

                    if not numeric_cols:
                        st.info("No matching numeric parameters found to visualize.")
                    else:
                        for col in numeric_cols:
                            st.markdown(f"#### 📊 {col} (Manual vs. WAMO)")
                            try:
                                merged = pd.merge(
                                    df[['date', col]],
                                    wamo_df[['date', col]],
                                    on='date',
                                    how='inner',
                                    suffixes=('_manual', '_wamo')
                                )

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=merged['date'], y=merged[f"{col}_manual"],
                                                         mode='lines+markers', name=f'Manual {col}'))
                                fig.add_trace(go.Scatter(x=merged['date'], y=merged[f"{col}_wamo"],
                                                         mode='lines+markers', name=f'WAMO {col}'))

                                fig.update_layout(
                                    title=f"{col} Over Time: Manual vs WAMO",
                                    xaxis_title='Date',
                                    yaxis_title=col,
                                    template='plotly_white'
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as inner_e:
                                st.warning(f"Could not plot {col}: {inner_e}")
                except Exception as parse_error:
                    st.warning(f"📅 Failed to convert date columns: {parse_error}")
                else:
                    st.warning("❌ Could not detect a valid date column in one of the datasets.")
            else:
                st.warning("⚠️ No WAMO data found for the selected lake.")
        else:
            st.info("Enable the checkbox in the sidebar to compare with WAMO data.")
    # Assistant Chat
    with tab4:
        st.subheader("💬 Ask the Assistant")
        user_prompt = st.chat_input("Ask a question or request a chart...")
        if user_prompt:
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
 f"You are a data assistant working with a Pandas DataFrame named `df`. "
 f"The user may ask questions or request charts. If they request a chart, return only Python code using Plotly Express (as `px`) in triple backticks. "
 f"Do not include explanations. "
 f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}.\n\n"
 f"Sample:\n{df.head(50).to_string(index=False)}"
                ),
                MessagesPlaceholder(variable_name="messages")
            ])

            response = (prompt | st.session_state["model"]).invoke({
                "messages": [HumanMessage(content=user_prompt)]
            })

            code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', response.content, re.DOTALL)
            chart_code = code_block_match.group(1).strip() if code_block_match else ""

            with st.chat_message("assistant"):
                if chart_code:
                    fig = get_fig_from_code(chart_code, df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("⚠️ Could not generate chart.")
                else:
                    st.markdown(response.content)

            st.session_state["messages"].append({"role": "assistant", "content": response.content})
else:
    st.info("📁 Upload a CSV or Excel file to begin.")
