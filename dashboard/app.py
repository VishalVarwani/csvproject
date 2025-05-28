import streamlit as st
import pandas as pd
import plotly.express as px
import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import io
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Load model once
if 'model' not in st.session_state:
    st.session_state['model'] = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

# Page setup
st.set_page_config(page_title="LLM Chart Generator", layout="wide")
st.title("📊 AI Chart Assistant (LangChain + Plotly + Streamlit)")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

# Sidebar for LLM control (optional)
st.sidebar.subheader("Model Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

# Message memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Plotly execution helper
def get_fig_from_code(code, df):
    local_vars = {
        "df": df.copy(),
        "px": px
    }
    try:
        exec(code, {}, local_vars)
        if "fig" not in local_vars:
            raise ValueError("fig not found in code")
        return local_vars["fig"]
    except Exception as e:
        st.error(f"⚠️ Error executing chart code: {e}")
        return None

# Main logic
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head(100))

    user_prompt = st.chat_input("Ask something about the data or request a chart...")

    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Extract context from df
        df_preview = df.head(5).to_string(index=False)
        column_list = ", ".join(df.columns.tolist())
        row_count, col_count = df.shape

        # Prompt setup
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are an expert data assistant working with a Pandas DataFrame named `df`. "
             f"The file uploaded is '{{name_of_file}}' with {row_count} rows and {col_count} columns. "
             f"The column names are: {column_list}. "
             f"Here are the first 5 rows:\n\n{df_preview}\n\n"
             f"You can either answer questions about the data or generate charts using Plotly (`plotly.express` as `px`). "
             f"Only return valid Python code inside triple backticks (```python ... ```), and only if the user asks for a chart. "
             f"For questions or analysis, answer in plain English. If unsure, explain politely.")
            ,
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | st.session_state["model"]

        response = chain.invoke({
            "messages": [HumanMessage(content=user_prompt)],
            "data": df_preview,
            "name_of_file": uploaded_file.name
        })

        # Handle response
        code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', response.content, re.DOTALL)
        chart_code = code_block_match.group(1).strip() if code_block_match else ""

        with st.chat_message("assistant"):
            if chart_code:
                chart_code = re.sub(r'(?m)^fig\.show\(\)\s*$', '', chart_code)
                fig = get_fig_from_code(chart_code, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("⚠️ I tried to generate a chart but ran into an error.")
            else:
                st.markdown(response.content)

        st.session_state["messages"].append({"role": "assistant", "content": response.content})

else:
    st.info("📂 Upload a CSV or Excel file to begin.")
