import joblib
from pathlib import Path
import gradio as gr


MODEL_DIR=Path().absolute().parent / "models"
MODEL_PATH=MODEL_DIR.joinpath("spamclassifier.pkl")

model_from_joblib = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

naive_bayes_from_joblib,count_vector = model_from_joblib


def classifier(user_input: str) -> str:
    user_input = count_vector.transform([user_input])
    user_output = naive_bayes_from_joblib.predict(user_input)[0]
    return 'spam' if user_output == 1 else 'ham'


# Gradio
def prediction(user_email):
    return classifier(user_email)

demo = gr.Interface(
    fn=prediction,
    inputs=["text"],
    outputs=["text"],
    title="Email Classifier"
)

demo.launch(share=True)