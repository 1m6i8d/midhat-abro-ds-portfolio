# Roman Urdu Sentiment Classifier

A fine-tuned multilingual BERT model that classifies Roman Urdu text into **Positive**, **Negative**, or **Neutral** sentiment.

---

## Project Overview

Roman Urdu is an informal writing system where Urdu is written using Latin script — widely used in Pakistani social media, SMS, and online communication. It has no standard spelling rules, making it a genuinely challenging NLP problem.

This project fine-tunes `bert-base-multilingual-cased` (mBERT) on a labeled Roman Urdu dataset to perform 3-class sentiment classification.

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 80.9% |
| Macro F1 | 80.9% |

**Per-class F1:**
- Negative: 0.82
- Neutral: 0.79
- Positive: 0.82

---

## Dataset

- **Source:** [HowMannyMore/romanurdu-sentiment-dataset](https://huggingface.co/datasets/HowMannyMore/romanurdu-sentiment-dataset)
- **Total samples:** 83,309
- **Splits:** Train (60,190) / Validation (10,622) / Test (12,497)
- **Classes:** Positive, Negative, Neutral (near-perfectly balanced)

---

## Model

- **Base model:** `bert-base-multilingual-cased`
- **Fine-tuning framework:** HuggingFace `transformers` Trainer API
- **Training environment:** Google Colab (T4 GPU)
- **Epochs:** 3
- **Batch size:** 32
- **Max token length:** 128

---

## How to Run

**1. Clone the repo and navigate to this project:**
```bash
cd projects/02-nlp-sentiment
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Download the model:**

The fine-tuned model is not included in this repo due to size (~700MB).
To reproduce it, run the notebook in `notebooks/roman_urdu_sentiment.ipynb` on Google Colab with a GPU runtime.
Place the downloaded model folder at `models/roman_urdu_sentiment_model/`.

**4. Run the app:**
```bash
streamlit run app/app.py
```

---

## Tech Stack

- Python 3.12
- HuggingFace `transformers` + `datasets`
- PyTorch
- Streamlit
- scikit-learn
- Google Colab (training)

---

## Sample Predictions

| Input | Prediction |
|---|---|
| `yeh film bohat achi thi` | Positive ✅ |
| `bohat bura hua yaar` | Negative ✅ |
| `kal mausam theek tha` | Neutral ✅ |