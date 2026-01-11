import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Demo model (trained nhanh)
@st.cache_resource
def load_demo_model():
    texts = [
        "This product is very good",
        "Excellent quality and fast delivery",
        "Bad product, very disappointed",
        "Terrible experience",
        "It is okay, not bad",
        "Average quality"
    ]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer

def show():
    st.markdown(
        "<h3 style='color:#2b6f3e;'>Analysis – Sentiment Analysis</h3>",
        unsafe_allow_html=True
    )

    st.write("Analyze product reviews and classify sentiment.")

    model, vectorizer = load_demo_model()

    # =========================
    # TEXT INPUT
    # =========================
    st.subheader("📝 Input Product Review")
    review = st.text_area(
        "Enter a product review (Vietnamese or English):",
        height=120
    )

    if st.button("▶️ Analyze Sentiment"):
        X = vectorizer.transform([review])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X).max()

        st.success(f"Predicted Sentiment: **{pred.upper()}**")
        st.info(f"Confidence: **{proba:.2f}**")

    # =========================
    # BATCH CSV
    # =========================
    st.write("---")
    st.subheader("📂 Upload Reviews Dataset (CSV)")

    file = st.file_uploader("CSV with column: review", type=["csv"])

    if file:
        df = pd.read_csv(file)
        X = vectorizer.transform(df["review"])
        df["sentiment"] = model.predict(X)

        st.dataframe(df.head())

        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot(
            kind="bar", ax=ax, color=["green", "gray", "red"]
        )
        st.pyplot(fig)
