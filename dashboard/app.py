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

# --- config ---
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# MongoDB
client = MongoClient(st.secrets["MONGO_URI"])
wamo_collection = client["Wamoproject"]["csv"]

# WAMO Mapping
WAMO_MAPPING = {
    "Babenhäuser See": "wamo00023",
    # add more
}

# Load model
if 'model' not in st.session_state:
    st.session_state['model'] = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

# --- helpers ---

def get_fig_from_code(code, df):
    local_vars = {"df": df.copy(), "px": px}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("fig")
    except Exception as e:
        st.error(f"Error generating chart: {e}")
        return None

def generate_comparison_prompt(df_manual, df_wamo):
    return f"""
You are a scientific data analyst. Two datasets are shown below:
- One is manually collected by researchers (CSV file uploaded)
- The other is WAMO sensor data for the same lake

Compare all parameters that appear to match — even if the column names are slightly different — and summarize their differences in a table like this:

| Parameter | Manual Avg | WAMO Avg | Difference | Trend |

Manual Sampling (first 5 rows):
{df_manual.head(5).to_string(index=False)}

WAMO Sensor (first 5 rows):
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
st.set_page_config(page_title="AI Chart Assistant", layout="wide")
st.title("📊 AI Chart Assistant")

with st.sidebar:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    selected_lake = st.selectbox("🏞️ Select Lake for WAMO Comparison", list(WAMO_MAPPING.keys()))
    compare_with_wamo = st.checkbox("🔄 Compare with WAMO", value=False)
    temperature = st.slider("🔥 Model Temperature", 0.0, 2.0, 0.7, 0.1)

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Preview Data", "📊 Suggested Charts", "📋 Compare with WAMO", "💬 Chat with Assistant"])

    with tab1:
        st.subheader("🔍 Preview Uploaded Data")
        st.dataframe(df.head(100))

    with tab2:
        st.subheader("📊 Auto-Generated Chart Suggestions")
        df_preview = df.head(100).to_string(index=False)
        suggestion_prompt = """
      You are a data visualization expert. The user has uploaded a dataset. Based on the first 100 rows, suggest 2 to 3 insightful visualizations using Plotly Express.
Only return valid Python code inside triple backticks like:
```python
fig = px.bar(df, x="col1", y="col2", title="Bar Chart")
Do not explain anything. Just return code blocks.
        """
        suggestion_response = st.session_state['model'].invoke([
            HumanMessage(content=suggestion_prompt + "\n\nCSV Preview:\n" + df_preview)
        ])
        code_blocks = re.findall(r"```(?:[Pp]ython)?(.*?)```", suggestion_response.content, re.DOTALL)
        for i, code in enumerate(code_blocks):
            fig = get_fig_from_code(code.strip(), df)
            if fig:
                st.subheader(f"Suggested Chart #{i+1}")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📋 LLM-Based Comparison with WAMO")
        if compare_with_wamo:
            wamo_df = fetch_wamo_df(selected_lake)
            if not wamo_df.empty:
                compare_prompt = generate_comparison_prompt(df, wamo_df)
                response = st.session_state["model"].invoke([HumanMessage(content=compare_prompt)])
                st.markdown(response.content)
            else:
                st.warning("No WAMO data found for selected lake.")
        else:
            st.info("Enable the checkbox in the sidebar to compare with WAMO data.")

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
     f"You are a data assistant working with a Pandas DataFrame called `df`. "
     f"The user may ask questions or request charts. If they request a chart, you must return only a Python code block using Plotly Express (as `px`) inside triple backticks. "
     f"Do NOT include explanations or non-code text when generating charts. "
     f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}. "
     f"First rows:\n\n{df.head(5).to_string(index=False)}"
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
    st.info("📁 Upload a CSV or Excel file to get started.")
