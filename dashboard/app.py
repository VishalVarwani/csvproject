
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from openai import OpenAI
import os
import re

# Setup
st.set_page_config(page_title="📊 CSV Chat Assistant", layout="wide")
client = OpenAI(api_key="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")

# Session State Setup
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Layout
col1, col2 = st.columns([3, 7])

# Chat Sidebar (left)
with col1:
    st.markdown("## 💬 Chat")
    uploaded_file = st.file_uploader("📎 Upload your CSV file", type=["csv"], key="uploader")

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.chat_history.append({"role": "system", "content": f"File '{uploaded_file.name}' uploaded successfully with shape {st.session_state.df.shape}"})

    user_input = st.chat_input("Ask a question about your data...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        if st.session_state.df is not None:
            def ask_llm(user_input, df):
                df_sample = df.head(100).to_csv(index=False)
                prompt = f"""
You are a data analysis assistant for researchers. The user has uploaded a CSV file. You can:
- Summarize the file
- Suggest meaningful visualizations
- Plot charts like line, scatter, histogram, bar
- Explain columns and patterns
- Detect nulls, outliers, or correlations

Here is a preview of the file (first 100 rows):
{df_sample}

Now respond to this user prompt:
"{user_input}"

If a chart is needed, respond only with one JSON block in this format:
{{"type": "scatter", "x": "col1", "y": "col2", "title": "Chart title"}}
Otherwise, reply in plain text only.
"""
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )
                return response.choices[0].message.content

            with st.spinner("Assistant is typing..."):
                response = ask_llm(user_input, st.session_state.df)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Chat Response (right)
with col2:
    st.markdown("## 🤖 Assistant Response")
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f"**You:** {chat['content']}")
            elif chat['role'] == 'system':
                st.success(chat['content'])
            elif chat['role'] == 'assistant':
                df = st.session_state.df
                # Try to extract JSON block using regex
                json_match = re.search(r'{.*}', chat['content'], re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        chart_type = parsed['type']
                        x, y, title = parsed.get('x'), parsed.get('y'), parsed.get('title')

                        if chart_type == 'scatter':
                            fig = px.scatter(df, x=x, y=y, title=title)
                        elif chart_type == 'line':
                            fig = px.line(df, x=x, y=y, title=title)
                        elif chart_type == 'bar':
                            fig = px.bar(df, x=x, y=y, title=title)
                        elif chart_type == 'histogram':
                            fig = px.histogram(df, x=x, title=title)
                        else:
                            st.warning("Unsupported chart type")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.markdown(f"**Assistant:** {chat['content']}")
                else:
                    st.markdown(f"**Assistant:** {chat['content']}")

    # Better scroll using JavaScript after response renders
    st.markdown("""
        <div id="bottom-scroll-anchor"></div>
        <script>
            document.getElementById('bottom-scroll-anchor').scrollIntoView({ behavior: 'smooth' });
        </script>
    """, unsafe_allow_html=True)
