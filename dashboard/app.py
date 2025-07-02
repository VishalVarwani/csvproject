# ----------------------------
# File: app.py
# ----------------------------
from tabs import tab_preview, tab_charts, tab_compare, tab_chat
from utils.mongo_utils import fetch_wamo_df
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
import os
import plotly.graph_objects as go


# Load secrets
load_dotenv()

# Init session state
if 'model' not in st.session_state:
    st.session_state['model'] = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model="llama3-70b-8192")

# MongoDB
client = MongoClient(st.secrets["MONGO_URI"])
wamo_collection = client["Wamoproject"]["CSV"]
upload_collection = client["Wamoproject"]["CSVUploads"]

# WAMO lake mapping
WAMO_MAPPING = {
    "Babenhäuser See": "wamo00023",
    # Add more mappings
}

# Streamlit Page Config
st.set_page_config(page_title="📊 AI Chart Assistant", layout="wide",initial_sidebar_state="expanded")

# Inject sidebar style
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 250px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 250px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 AI Chart Assistant")

# ---------------------------
# Utility Functions
# ---------------------------

def save_uploaded_file(file, df):
    filename = file.name
    existing = upload_collection.find_one({"filename": filename})
    if not existing:
        upload_collection.insert_one({
            "filename": filename,
            "uploaded_at": datetime.datetime.utcnow(),
            "data": df.to_dict("records")
        })

def get_uploaded_filenames():
    return [doc["filename"] for doc in upload_collection.find({}, {"filename": 1})]

def load_file_from_mongo(filename):
    doc = upload_collection.find_one({"filename": filename})
    if doc:
        return pd.DataFrame(doc["data"])
    return None

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    selected_lake = st.selectbox("🏞️ Select Lake for WAMO Comparison", list(WAMO_MAPPING.keys()))

    st.markdown("---")
    st.markdown("📚 **Previously Uploaded Files**")
    prev_files = get_uploaded_filenames()
    dropdown_options = ["📂 Select a file..."] + prev_files if prev_files else ["No files yet"]
    selected_prev_file = st.selectbox("🕘 Choose Existing File", dropdown_options)

# Block data load unless user explicitly chooses a file
    df = None
    if selected_prev_file not in ["📂 Select a file...", "No files yet"]:
        df = load_file_from_mongo(selected_prev_file)

# ---------------------------
# Upload Handling (save only)
# ---------------------------

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            temp_df = pd.read_csv(uploaded_file)
        else:
            temp_df = pd.read_excel(uploaded_file)

        save_uploaded_file(uploaded_file, temp_df)
        st.success(f"✅ '{uploaded_file.name}' uploaded successfully.")
        st.info("📌 Please select the uploaded file from the dropdown below to view it.")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

# ---------------------------
# Load Selected File (user-driven)
# ---------------------------

df = None
if selected_prev_file and selected_prev_file != "No files yet":
    df = load_file_from_mongo(selected_prev_file)

# ---------------------------
# Main Tabs UI
# ---------------------------

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Preview Data", "📊 Suggested Charts", "📋 Compare with WAMO", "💬 Chat with Assistant"])

    with tab1:
        tab_preview.render(df)

    with tab2:
        tab_charts.render(df)

    with tab3:
        tab_compare.render(df, selected_lake, WAMO_MAPPING, wamo_collection)

    with tab4:
        tab_chat.render(df)
else:
    st.info("📁 Please select a file from the dropdown or upload a new one.")
