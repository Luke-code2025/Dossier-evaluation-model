import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("dossier_model.joblib")
features = joblib.load("feature_list.joblib")


# Set the page title and layout
st.set_page_config(page_title="Your Drug Dossier Evaluation AI-powered Assistant")

# App header
st.title("Your Drug Dossier Evaluation AI-powered Assistant")
st.markdown("""
Welcome to your AI-powered assistant for evaluating drug dossiers. 
Upload a dossier checklist in `.csv` format and let the model predict its **regulatory quality outcome**.
""")

# Upload CSV file
uploaded_file = st.file_uploader(" Upload your dossier checklist CSV")

if uploaded_file:
    try:
        # Load model and features
        model = joblib.load("dossier_model.joblib")
        feature_list = joblib.load("feature_list.joblib")  # This must match the training feature set

        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Check for required features
        missing_features = [col for col in feature_list if col not in df.columns]
        if missing_features:
            st.error(f"❌ Missing required columns: {missing_features}")
        else:
            # Make predictions
            prediction = model.predict(df[feature_list])
            prediction_label = ['❌ Not Accepted' if p == 0 else '✅ Accepted' for p in prediction]

        
            # Show result
            st.success("✅ Dossier evaluation completed.")
            st.dataframe(df)

            # Downloadable result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Evaluation Results",
                data=csv,
                file_name="dossier_evaluation_result.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("📌 Please upload a CSV file to begin the evaluation.")

# Footer
st.markdown("---")
st.markdown("🔒 All predictions are local and confidential. This tool is for evaluation support only.")
