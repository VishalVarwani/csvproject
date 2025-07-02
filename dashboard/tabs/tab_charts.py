
# ----------------------------
# File: tabs/tab_charts.py
# ----------------------------
import streamlit as st
import re
import plotly.express as px
import pandas as pd
from utils.chart_executor import get_fig_from_code
from langchain_core.messages import HumanMessage

def render(df):
    st.subheader("📊 Chart Exploration")

    # Toggle: AI vs Manual
    mode = st.radio(
        "How would you like to explore the charts?",
        ["AI Suggestions", "Manual Chart Builder"],
        horizontal=True
    )

    if mode == "AI Suggestions":
        if "chart_suggestions" not in st.session_state:
            df_preview = df.head(100).to_string(index=False)
            column_list = ", ".join(df.columns.tolist())

            # Dynamically detect appropriate example x and y
            datetime_col = next((col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower()), None)
            numeric_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)

            example_x = datetime_col if datetime_col else df.columns[0]
            example_y = numeric_col if numeric_col else df.columns[1]

            example_code = f"""```python
fig = px.line(df, x="{example_x}", y="{example_y}", title="{example_y} over {example_x}")
```"""

            suggestion_prompt = f"""
You are an AI assistant helping a water scientist explore a water quality dataset.

🎯 Based on the first 100 rows of this dataset, suggest exactly **3 insightful and relevant charts** using Plotly Express.

✅ Only use the columns listed below exactly as they are:
{column_list}

⚠️ Do not invent or assume any column names.

Respond only with Python code blocks like:

{example_code}

Requirements:
- Always return exactly 3 suggestions
- Use **different chart types** where possible
- Avoid generic or repeated chart types
- Use columns present in the dataset

Data Preview:
{df_preview}
"""

            response = st.session_state["model"].invoke([
                HumanMessage(content=suggestion_prompt)
            ])

            code_blocks = re.findall(r"```(?:[Pp]ython)?(.*?)```", response.content, re.DOTALL)[:3]
            st.session_state["chart_suggestions"] = code_blocks

        # Render AI-suggested charts with collapsible insights
        for i, code in enumerate(st.session_state["chart_suggestions"]):
            fig = get_fig_from_code(code.strip(), df)
            if fig:
                st.subheader(f"Suggested Chart #{i + 1}")
                st.plotly_chart(fig, use_container_width=True)

                # Generate insight explanation
                df_preview = df.head(100).to_string(index=False)
                explanation_prompt = f"""
You are a water quality researcher. Below is a chart generated from a dataset.

The chart was created using this code:
```python
{code.strip()}
```

The first 100 rows of the DataFrame `df` look like this:
{df_preview}

Please provide an insightful explanation of what the chart shows:
- Focus on patterns, anomalies, or relationships in the data
- Avoid explaining the code or chart type
- Keep it relevant to environmental water analysis
"""

                insight = st.session_state["model"].invoke([
                    HumanMessage(content=explanation_prompt)
                ])

                with st.expander("🧠 Chart Insight"):
                    st.markdown(insight.content)

    elif mode == "Manual Chart Builder":
        st.subheader("🎛️ Build Your Own Chart")

        with st.form("manual_chart_form"):
            columns = df.columns.tolist()
            x_axis = st.selectbox("📈 X-axis", columns)
            y_axis = st.selectbox("📉 Y-axis", columns)
            chart_type = st.selectbox("📊 Chart Type", ["Line", "Scatter", "Bar"])
            submit = st.form_submit_button("Generate Chart")
        if submit:
            try:
                if chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                else:
                    st.warning("Unsupported chart type.")
                    return
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Error generating manual chart: {e}")
                if st.button("🔁 Retry Generating Chart", key=f"retry_button_{hash(code)}"):
                    st.rerun()
