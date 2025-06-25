import streamlit as st
import re
from utils.chart_executor import get_fig_from_code
from langchain_core.messages import HumanMessage

def render(df):
    st.subheader("📊 AI-Suggested Charts")
    df_preview = df.head(100).to_string(index=False)
    suggestion_prompt = """
You are a data visualization expert. Based on the first 100 rows of this dataset, suggest 2–3 insightful Plotly Express visualizations.
Return valid code blocks only (no explanations), like:
```python
fig = px.line(df, x=\"Time\", y=\"pH\", title=\"pH over Time\")
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