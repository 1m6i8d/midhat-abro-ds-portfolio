import os
import json
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# configure
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "roman_urdu_sentiment_model"
)

# load model (cached)
@st.cache_resource
def load_model():
    mappings_path = os.path.join(MODEL_PATH, "label_mappings.json")
    with open(mappings_path, "r") as f:
        mappings = json.load(f)

    id2label = {int(k): v for k, v in mappings["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()
    return tokenizer, model, id2label

# predict
def predict_sentiment(text, tokenizer, model, id2label):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    pred_id = torch.argmax(probs).item()
    pred_label = id2label[pred_id]

    prob_dict = {
        id2label[i]: round(probs[i].item() * 100, 2)
        for i in range(len(id2label))
    }
    return pred_label, prob_dict

# streamlit UI

st.set_page_config(
    page_title="Roman Urdu Sentiment Classifier",
    layout="centered",
)

st.markdown(
"""
    <style>
    .stApp {
        background-color: #ddf1f8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Roman Urdu Sentiment Classifier")
st.markdown(
    "Type any sentence in **Roman Urdu** and the model will predict"
    "whether the sentiment is **Positive**, **Negative**, or **Neutral**."
)

with st.spinner("Loading model (may take a while)..."):
    tokenizer, model, id2label = load_model()

st.success("Model successfully loaded.")

st.subheader("Enter Roman Urdu Text")
user_input = st.text_area(
    label="",
    placeholder="e.g. yeh movie maze ki thi...",
    height=120
)

if st.button("Analyse Sentiment", type="primary"):
    if not user_input.strip(): # checks empty input
        st.warning("Please enter some text before analysing.")
    else:
        with st.spinner("Analysing..."):
            label, probs = predict_sentiment(user_input, tokenizer, model, id2label)

        colour_map = {
            "Positive": "green",
            "Neutral": "orange",
            "Negative": "red",
        }
        colour = colour_map[label]

        st.subheader("Result")
        st.markdown(
            f"<h2 style='color:{colour};'>{label}</h2>",
            unsafe_allow_html=True
        )
        # display conf scores as bar chart
        st.subheader("Confidence Scores")
        st.bar_chart(probs)

        st.subheader("Breakdown")
        for sentiment, percentage in sorted(probs.items(), key=lambda x: -x[1]):
            st.write(f"**{sentiment}:** {percentage}%")

# Footer
st.caption(
    "Model: bert-base-multilingual-cased fine-tuned on "
    "HowMannyMore/romanurdu-sentiment-dataset · Built with HuggingFace + Streamlit"
)