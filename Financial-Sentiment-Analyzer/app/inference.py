import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODEL_DIR = r"D:\TransfersFile\Singh\OneDrive\Desktop\AIML_Project\Financial-Sentiment-Analyzer\models\fin_lora_adapter"


class FinancialSentimentModel:
    def __init__(self):
        base_model_name = "yiyanghkust/finbert-tone"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)

        self.model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        self.model.eval()

        self.id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        pred = probs.argmax()
        confidence = float(probs[pred] * 100)

        return self.id2label[pred], confidence
