import streamlit as st
import os
import pandas as pd
import plotly.express as px
import json
import re
from openai import OpenAI

# Page Title
st.title("OpenAI ChatGPT + CSV Assistant")

# Session Setup
if 'model' not in st.session_state:
    st.session_state['model'] = OpenAI(api_key="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")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'df' not in st.session_state:
    st.session_state['df'] = None



# CSV Upload Section
st.sidebar.title("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
   if uploaded_file and 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded_file.name:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state['last_uploaded'] = uploaded_file.name
    st.session_state['messages'].append({
        "role": "system",
        "content": f"File '{uploaded_file.name}' uploaded with shape {st.session_state.df.shape}"
    })


# Display previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        # Try to render plot if assistant message has JSON
        if message['role'] == 'assistant' and st.session_state.df is not None:
            match = re.search(r'{.*}', message['content'], re.DOTALL)
            if match:
                try:
                    chart = json.loads(match.group())
                    chart_type = chart.get("type")
                    x = chart.get("x")
                    y = chart.get("y")
                    title = chart.get("title")
                    df = st.session_state.df

                    if chart_type == "bar":
                        fig = px.bar(df, x=x, y=y, title=title)
                    elif chart_type == "line":
                        fig = px.line(df, x=x, y=y, title=title)
                    elif chart_type == "scatter":
                        fig = px.scatter(df, x=x, y=y, title=title)
                    elif chart_type == "histogram":
                        fig = px.histogram(df, x=x, title=title)
                    else:
                        st.warning("Unsupported chart type")
                    st.plotly_chart(fig, use_container_width=True)
                    continue  # skip markdown if plot rendered
                except:
                    pass
        st.markdown(message['content'])

# Input prompt
if prompt := st.chat_input("Ask a question about your CSV file or anything else"):
    # Show user message
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create GPT Prompt
    df_preview = st.session_state.df.head(50).iloc[:, :4].to_csv(index=False) if st.session_state.df is not None else ""
    prompt_wrapper = f"""
You are a data analysis assistant. If a CSV file is uploaded, use it to:
- Summarize the dataset
- Suggest visualizations (e.g., histogram, scatter, bar, line)
- Identify patterns, nulls, outliers, or stats
- Explain columns and relationships

Respond in plain text. If a chart is needed, return exactly one JSON object like:
{{"type": "scatter", "x": "col1", "y": "col2", "title": "My Chart"}}

CSV Preview:
{df_preview}

User Request:
{prompt}
"""

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        client = st.session_state['model']
        history = st.session_state['messages'][-2:]
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state['messages']][-2:] + [{"role": "user", "content": prompt_wrapper}],
            
            stream=True
        )
        response = st.write_stream(stream)

    # Store assistant reply
    st.session_state['messages'].append({"role": "assistant", "content": response})
