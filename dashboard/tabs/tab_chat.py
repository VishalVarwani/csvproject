# ----------------------------
# File: tabs/tab_chat.py
# ----------------------------
import streamlit as st
import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from utils.chart_executor import get_fig_from_code


def render(df):
    st.subheader("💬 Ask the Assistant")

    # Chat Input
    user_prompt = st.chat_input("Ask a question or request a chart...")
    if user_prompt:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Append user message
        st.session_state["messages"].append(HumanMessage(content=user_prompt))

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Construct prompt with detailed system message
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful data assistant for a scientist analyzing water quality data using a pandas DataFrame named `df`. 
The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns. The columns are: {', '.join(df.columns)}.
Here is a sample of the data:\n\n{df.head(50).to_string(index=False)}

You should:
1. Think aloud before showing code.
2. Generate **Python code with Plotly Express** using the DataFrame `df`.
3. Always wrap code inside triple backticks ```python ... ``` so the app can extract it.

If a question is ambiguous, ask follow-up questions before proceeding.
"""),
            MessagesPlaceholder(variable_name="messages")
        ])

        # Get response from model
        response = (prompt | st.session_state["model"]).invoke({"messages": st.session_state["messages"]})

        # Display assistant reply
        with st.chat_message("assistant"):
            st.session_state["messages"].append(response)

            # Extract chart code
            code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', response.content, re.DOTALL)
            chart_code = code_block_match.group(1).strip() if code_block_match else ""

            if chart_code:
                fig = get_fig_from_code(chart_code, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("✅ Chart generated successfully.")
                else:
                    st.markdown("⚠️ Chart code found, but something went wrong. Try checking if `fig = px...` is correctly defined.")
            else:
                st.markdown(response.content)
