import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("dossier_model.joblib")
features = joblib.load("feature_list.joblib")

st.title("💊 Drug Dossier Evaluation Assistant")
st.write("Upload a CSV file with dossier features (0 = Absent, 1 = Present)")

uploaded_file = st.file_uploader("Upload dossier file (CSV)", type="csv")

if uploaded_file:
    user_data = pd.read_csv(uploaded_file)

    # Ensure the feature order matches the model
    try:
        user_data = user_data[features]
        prediction = model.predict(user_data)
        prediction_proba = model.predict_proba(user_data)[:, 1]

        user_data["Evaluation_Outcome"] = prediction
        user_data["Probability (Acceptable)"] = prediction_proba

        st.success("✅ Dossier evaluated!")
        st.write(user_data)

        csv = user_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", data=csv, file_name="evaluation_results.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Error: {e}")
# Upload and read the sample dossier
uploaded_file = st.file_uploader("Upload dossier for evaluation", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load model and features
    model = joblib.load("your_model.joblib")
    feature_list = joblib.load("feature_list.joblib")  # list of features used in training

    # Ensure required features are present
    missing = [col for col in feature_list if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
    else:
        # Make predictions
        prediction = model.predict(df[feature_list])

        # Add predictions to the dataframe
        df['Evaluation_Outcome'] = prediction

        st.success("✅ Evaluation complete")
        st.dataframe(df)
