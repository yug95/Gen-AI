import streamlit as st
import os
import pandas as pd

# Define directory to save uploaded files
UPLOAD_DIR = "/Users/yogeshagrawal/Desktop/Gen AI/32.Agentic_AI/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
st.title("📁 Upload Multiple CSV Files and Save to Folder")

# Upload multiple CSVs
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        # Define file path
        file_path = os.path.join(UPLOAD_DIR, file.name)
        
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        st.success(f"✅ Saved: `{file.name}` to `{UPLOAD_DIR}`")

        # Optional: Read and show preview
        try:
            df = pd.read_csv(file_path)
            st.subheader(f"📄 Preview: {file.name}")
            st.dataframe(df)
        except Exception as e:
            st.error(f"⚠️ Could not read {file.name}: {e}")