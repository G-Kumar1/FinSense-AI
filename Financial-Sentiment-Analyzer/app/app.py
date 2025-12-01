import streamlit as st
from inference import FinancialSentimentModel
from news_fetcher import get_financial_news

st.set_page_config(page_title="Financial Sentiment Analyzer", layout="wide")

model = FinancialSentimentModel()

st.title("📊 Financial News Sentiment Analyzer")
st.write("Analyze financial news sentiment using LoRA-fine-tuned FinBERT")

option = st.radio(
    "Choose input mode:",
    ("Manual Input", "Fetch Latest Financial News Automatically")
)

if option == "Manual Input":
    user_input = st.text_area("Paste financial news text here:")

    if st.button("Analyze"):
        if user_input.strip():
            label, confidence = model.predict(user_input)
            st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")
        else:
            st.warning("Please enter text.")

else:
    if st.button("Fetch & Analyze News"):
        news_list = get_financial_news()
        results = []

        for news in news_list:
            label, confidence = model.predict(news)
            results.append((news, label, confidence))

        for news, label, conf in results:
            st.write(f"### 📰 {news}")
            st.write(f"**Sentiment:** {label} ({conf:.2f}%)")
            st.write("---")
