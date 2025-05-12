import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import clean_text
import joblib

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load data
df = pd.read_csv('data/emotion_dataset.csv')
df['text'] = df['text'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])
joblib.dump(le, 'label_encoder.pkl')

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.1)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = EmotionDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=64)
val_dataset = EmotionDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=64)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train
model.train()
for epoch in range(2):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# Save model
model.save_pretrained("bert_emotion_model")
tokenizer.save_pretrained("bert_emotion_model")
