import streamlit as st
import joblib
model=joblib.load("fake_news_svm.pkl")
vectorizer=joblib.load("tfidf_vectorizer.pkl")
st.title("📰 Fake News Detection")

st.write("Enter a news article below to check whether it is Fake or Real.")

news_text = st.text_area(
    "Enter News Article:",
    height=250
)
if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")

    else:
        text_tfidf = vectorizer.transform([news_text])

        prediction = model.predict(text_tfidf)[0]

        if prediction == 0:
            st.error("🚨 This news is likely FAKE")
        else:
            st.success("✅ This news is likely REAL")