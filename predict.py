import torch
from transformers import BertTokenizer, BertForSequenceClassification
from preprocess import clean_text
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained("bert_emotion_model")
tokenizer = BertTokenizer.from_pretrained("bert_emotion_model")
model.to(device)
model.eval()

le = joblib.load('label_encoder.pkl')

def predict_emotion(text):
    text = clean_text(text)
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="max_length"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return le.inverse_transform([predicted_class_id])[0]

if __name__ == "__main__":
    while True:
        text = input("Nhập câu (hoặc gõ 'exit'): ")
        if text.lower() == 'exit':
            break
        print("👉 Cảm xúc dự đoán:", predict_emotion(text))
