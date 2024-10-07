import joblib
from pathlib import Path
import streamlit as st


MODEL_DIR=Path().absolute().parent / "models"
MODEL_PATH=MODEL_DIR.joinpath("spamclassifier.pkl")

model_from_joblib = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

naive_bayes_from_joblib,count_vector = model_from_joblib


def classifier(user_input: str) -> str:
    user_input = count_vector.transform([user_input])
    user_output = naive_bayes_from_joblib.predict(user_input)[0]
    return 'spam' if user_output == 1 else 'ham'


# Streamlit
st.title("Email Classifier")
user_email = st.text_area("Enter the email :")

if st.button("Predict"):
    prediction = classifier(user_email)
    st.write('The given email is :{} \n'.format(prediction))