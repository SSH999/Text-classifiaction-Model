# Load the dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np 
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dataset = load_dataset('ag_news')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the text data
import re
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

def preprocess(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    return text

dataset = dataset.map(lambda examples: {'text': preprocess(examples['text']), 'label': examples['label']})

# Fine-tune the pre-trained model using Hugging Face library
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

# Train the model and evaluate its performance
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

print(eval_result)

test_text = [
    "The stock market was up today on positive news about the economy.",
    "The new restaurant in town has amazing food.",
    "The latest movie by Quentin Tarantino was disappointing.",
    "The football team won the championship game yesterday."
]

encoded_test_text = tokenizer(test_text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    output = model(encoded_test_text['input_ids'], attention_mask=encoded_test_text['attention_mask'])

print(output)
