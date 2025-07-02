
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
import plotly.figure_factory as ff
import numpy as np
def render(df):
    st.subheader("🔍 Preview Uploaded Data")

    # 🧠 AI LLM Summary
    with st.expander("🧠 AI Summary of Your Dataset", expanded=True):
        try:
            df_preview = df.head(5).to_string(index=False)
            missing_summary = df.isnull().mean().round(3) * 100
            missing_str = missing_summary.to_string()
            numeric_df = df.select_dtypes(include='number')
            stats = numeric_df.describe().T[['mean', 'std', 'min', 'max']].to_string()

            summary_prompt =  f"""
You are a senior environmental scientist advising on water quality in the EU region. You are analyzing data collected from a freshwater monitoring station, loaded as a pandas DataFrame named `df`.

The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

🧪 Sample (first 5 rows):
{df_preview}

📉 Missing values (% per column):
{missing_str}

📊 Numeric statistics (mean, std, min, max):
{stats}

Write a detailed, expert-level summary structured as follows:

---

🔍 **Part 1: Analytical Bullet Points**  
Write 6–10 insights that reflect domain expertise. Include:
- Parameter interpretation (e.g., what DO, pH, turbidity, etc. mean)
- Any **flatlined sensors** or low-variance columns
- **High missingness** (especially if >30%)
- **Suspicious or biologically implausible values**, such as:
  - `Dissolved Oxygen (DO)` < **5 mg/L** → may indicate hypoxic or eutrophic conditions (EU threshold)
  - `pH` outside **6.5–8.5** → may harm aquatic life (EU guideline)
  - `Temperature` < **0°C** or > **35°C** → likely incorrect or harmful
  - `Turbidity` > **5 NTU** → indicates potential sediment or pollution (drinking threshold)
- **Timestamps**: is the data evenly spaced? Are there any large time gaps or data drop-offs?
- Mention if values are **clustered**, constant, or extremely skewed
- Suggest if values are **trustworthy** or need sensor/data validation

---

📘 **Part 2: Executive Summary Paragraph**  
Summarize the dataset’s overall health and fitness for analysis:
- Highlight **general usability**, **risks**, and **recommendations**
- Suggest next steps like: removing outliers, imputing values, excluding faulty parameters, or reviewing station logs
- Write professionally — imagine this being sent to a water policy taskforce or quality assessment board

Avoid generic statements like “There are X rows and Y columns.” Your goal is to help scientists and EU-level regulators make informed decisions based on this data.
"""
            response = st.session_state["model"].invoke([HumanMessage(content=summary_prompt)])
            st.markdown(response.content)
        except Exception as e:
            st.warning(f"⚠️ Could not generate LLM summary: {e}")

    # 📋 First 100 rows
    with st.expander("🔍 View First 100 Rows"):
        st.dataframe(df.head(100))
        st.markdown(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")

    # ❓ Missing value summary
    with st.expander("❓ Missing Value Summary", expanded=False):
        # 🔹 Column-level summary
        st.markdown("**🔹 Missing Count per Column:**")
        st.dataframe(
            df.isnull().sum()
            .reset_index()
            .rename(columns={"index": "Column", 0: "Missing Count"})
        )

        # 🔍 Row-level missing value detail
        st.markdown("**🔍 Detailed Missing Value Locations:**")
        missing_locations = df.isnull().stack().reset_index()
        missing_locations.columns = ["Row Index", "Column", "Is Missing"]
        missing_locations = missing_locations[missing_locations["Is Missing"] == True].drop("Is Missing", axis=1)

        if not missing_locations.empty:
            col_filter = st.selectbox(
                "🔽 Filter missing values by column:",
                ["All"] + sorted(missing_locations["Column"].unique().tolist()),
                key="missing_filter"
            )
            if col_filter != "All":
                filtered = missing_locations[missing_locations["Column"] == col_filter]
                st.dataframe(filtered)
            else:
                st.dataframe(missing_locations)
        else:
            st.success("✅ No missing values found in the dataset.")

    # 🔠 Column types
    # with st.expander("🔠 Column Data Types"):
    #     st.dataframe(
    #         df.dtypes.astype(str)
    #         .reset_index()
    #         .rename(columns={"index": "Column", 0: "Data Type"})
    #     )

    # 📊 Numeric summary statistics
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        with st.expander("📊 Numeric Summary Statistics", expanded=False):
            stats = numeric_df.describe().T
            missing_percent = df[numeric_df.columns].isnull().mean() * 100

            summary = stats[['mean', 'std', 'min', 'max']].copy()
            summary['% Missing'] = missing_percent
            summary = summary.reset_index().rename(columns={'index': 'Column'})
            st.dataframe(summary)

            # 🔔 Bell Curve Visualization
            st.markdown("### 📈 Distribution (Bell Curve)")
            selected_col = st.selectbox("Select a numeric column to view distribution:", numeric_df.columns)
            selected_data = df[selected_col].dropna().values.tolist()
            if len(selected_data) > 1:
                hist_data = [selected_data]
                group_labels = [selected_col]
                custom_colors = ['#8ab6f9']  # soft blue

                fig = ff.create_distplot(
                    hist_data,
                    group_labels,
                    show_hist=True,   # Histogram
                    show_rug=True,    # Rug plot (small vertical lines)
                    show_curve=True,
                    bin_size=0.2,
                    curve_type='kde',
                    colors=custom_colors,
  # KDE line (bell curve)
                )
                fig.update_layout(
                    title=f"Distribution of {selected_col}",
                    xaxis_title=selected_col,
                    yaxis_title="Density",
                    template="plotly_white",
                   
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        )
                 )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Not enough data to plot a distribution.")
    else:
        st.info("No numeric columns found for summary statistics.")


    # 🚨 Outlier detection
    if not numeric_df.empty:
        with st.expander("🚨 Outlier Detection Over Time"):
            date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            if date_cols:
                date_col = st.selectbox("Select a date/time column", date_cols)
                param_col = st.selectbox("Select a numeric column to visualize", numeric_df.columns)

                df_copy = df[[date_col, param_col]].dropna()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_copy = df_copy.dropna()

                Q1 = df_copy[param_col].quantile(0.25)
                Q3 = df_copy[param_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_copy["is_outlier"] = (
                    (df_copy[param_col] < lower_bound) | (df_copy[param_col] > upper_bound)
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_copy[date_col],
                    y=df_copy[param_col],
                    mode='lines+markers',
                    name="Normal Values",
                    marker=dict(color="gray"),
                    line=dict(color="lightgray")
                ))
                fig.add_trace(go.Scatter(
                    x=df_copy[df_copy["is_outlier"]][date_col],
                    y=df_copy[df_copy["is_outlier"]][param_col],
                    mode='markers',
                    name="Possible Outliers",
                    marker=dict(color="red", size=10, symbol="circle-open")
                ))
                fig.update_layout(
                    title=f"{param_col} Over Time (Outliers Highlighted)",
                    xaxis_title=date_col,
                    yaxis_title=param_col,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No date/time column found. Add one to enable this chart.")
