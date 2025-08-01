# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed guideline data (already combined PPB and WHO)
@st.cache_data
def load_guidelines():
    ppbwho_df = pd.read_csv("ppbwho_combined.csv")  # Make sure this file exists in your environment
    return ppbwho_df

def embed_text(corpus):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return vectorizer, vectorizer.fit_transform(corpus)

def compute_similarity(input_text, guideline_matrix, vectorizer):
    input_vec = vectorizer.transform([input_text])
    scores = cosine_similarity(input_vec, guideline_matrix)
    return scores.mean()

# Streamlit UI
st.set_page_config(page_title="Drug Dossier Evaluation AI", layout="centered")
st.title("Drug Dossier Evaluator")
st.markdown("Upload a drug dossier summary to evaluate it against regulatory guidelines.")

ppbwho_df = load_guidelines()
corpus = ppbwho_df['text'].dropna().values
vectorizer, guideline_matrix = embed_text(corpus)

# Upload and evaluate dossier
uploaded_file = st.file_uploader("Upload dossier text file", type=["txt"])

if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Dossier Text", input_text, height=200)
    score = compute_similarity(input_text, guideline_matrix, vectorizer)
    st.write(f"\n**Similarity Score:** {score:.2f}")

    if score > 0.3:
        st.success("✅ This dossier is aligned with guideline principles. Recommend approval.")
    elif score > 0.15:
        st.warning("⚠️ Partial alignment. Needs detailed review.")
    else:
        st.error("❌ Low alignment with guidelines. Not recommended in current form.")
