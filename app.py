import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Your Drug Dossier Evaluation AI-powered Assistant",
    layout="wide",
)

# --- Header ---
st.title("Your Drug Dossier Evaluation AI-powered Assistant")
st.markdown("""
Welcome to your AI-powered assistant for evaluating drug dossiers.  
Upload a dossier checklist in **`.csv`** format and let the model predict its **regulatory outcome**.
""")

st.markdown("---")

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload your dossier checklist CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("🔍 Evaluating dossier..."):
        try:
            # Load model and feature list
            model = joblib.load("dossier_model.joblib")
            feature_list = joblib.load("feature_list.joblib")

            # Read uploaded file
            df = pd.read_csv(uploaded_file)

            # Validate required features
            missing_features = [col for col in feature_list if col not in df.columns]
            if missing_features:
                st.error(f"❌ Missing required columns: {missing_features}")
                st.stop()

            # Predict outcomes
            predictions = model.predict(df[feature_list])
            df['Evaluation_Outcome'] = ['✅ Accepted' if p == 1 else '❌ Not Accepted' for p in predictions]

            # Display results directly
            st.success("🎯 Evaluation completed below:")
            st.dataframe(df[['Evaluation_Outcome'] + feature_list], use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred while processing the file:\n\n{e}")
else:
    st.info("📌 Please upload a `.csv` file to begin evaluation.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
    🔒 All evaluations are processed locally. Your data is private and never stored.<br>
    ⚠️ This tool is intended for support only. Always refer to official regulatory guidelines.
</div>
""", unsafe_allow_html=True)
st.write("✅ Loaded feature list:", feature_list.joblib)