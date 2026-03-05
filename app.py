import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words =  set(stopwords.words("english"))

# Load model and vectorizer
model = pickle.load(open("complaint_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Clean text function
def clean_text(text):
    if not text or text is None:
        return ""
    text =str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [ w for w in words if w not in stop_words]
    return " ".join(words)


# Page Config
st.set_page_config(
    page_title="Complaint Classifier",
    page_icon=" ",
    layout="centered"
)

# Title
st.title("Financial Complaint Classifier")
st.write("Enter a customer complaint to predict the product category")

# Input
user_input = st.text_area("Enter complaint here:", height=150, placeholder="e.g. There is an error on my credit report...")


# Classify button
if st.button("Classify Complaint"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        cleaned = clean_text(user_input)
        if cleaned.strip() == "":
            st.warning("Please enter a more descriptive complaint.")
        else:
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized).max() * 100
            st.success(f" **Product Category:** {prediction.replace('_', ' ').title()}")
            st.info(f" **Confidence:** {confidence:.1f}%")
            st.progress(int(confidence))
        
    

# Sidebar
st.sidebar.title("About")
st.sidebar.write("This app classifies financial complaints into:")
for category in ["Credit Card", "Credit Reporting", "Debt Collection", "Mortages and Loans", "Retail Banking"]:
    st.sidebar.write(f".{category}")
st.sidebar.write("---")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Accuracy: 87%")

