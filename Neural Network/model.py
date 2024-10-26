import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Define constants
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 2e-5
DROPOUT_PROB = 0.3  # Adjusted for better regularization
PATIENCE = 3  # Early stopping patience
WEIGHT_DECAY = 1e-2  # Added for regularization

# Load and preprocess the dataset
df = pd.read_csv('skill.csv')

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define a custom dataset class
class ResumeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, mlb, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = mlb.transform(dataframe['Category'].apply(lambda x: [x]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        resume = str(self.data.iloc[idx]['Resume'])
        inputs = self.tokenizer(resume, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return inputs, labels

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=102, output_hidden_states=False)  # Removed dropout from here

# Manually adjust dropout for attention layers
for layer in model.roberta.encoder.layer:
    layer.attention.self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
    layer.output.dropout = torch.nn.Dropout(p=DROPOUT_PROB)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# MultiLabelBinarizer for transforming categories
mlb = MultiLabelBinarizer()
mlb.fit(df['Category'].apply(lambda x: [x]))

# Create train and validation datasets and loaders
train_dataset = ResumeDataset(train_df, tokenizer, mlb)
val_dataset = ResumeDataset(val_df, tokenizer, mlb)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Define optimizer with weight decay for regularization
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Gradient scaler for mixed precision training
scaler = GradScaler()

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()

        # Mixed precision forward and backward pass
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            val_loss += outputs.loss.item()

            # Collect predictions and true labels for metric calculation
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.append((preds > 0.5).astype(int))
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate validation metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    val_recall = recall_score(all_labels, all_preds, average='weighted')

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        model.save_pretrained('./best_model')
        tokenizer.save_pretrained('./best_tokenizer')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Save the final model
model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_tokenizer')

# Define the prediction function
def predict(resume, model, mlb, tokenizer, device):
    model.eval()
    with torch.no_grad():
        # Tokenize the input resume
        inputs = tokenizer(resume, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        # Get predictions (logits) from the model
        outputs = model(inputs['input_ids'], attention_mask=(inputs['input_ids'] != 0).long().to(device))

        # Convert logits to probabilities using sigmoid function (for multi-label classification)
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

        # Get the predicted categories using the threshold (probability > 0.5)
        predictions = (probabilities > 0.05).astype(int)

        # Convert predictions back to categories using the MultiLabelBinarizer
        predicted_labels = mlb.inverse_transform(np.array([predictions]))

        # Print raw probabilities along with categories
        category_probabilities = {mlb.classes_[i]: probabilities[i] for i in range(len(probabilities))}

        return predicted_labels, category_probabilities

# Example resumes for testing the prediction
test_resumes = [
    "Experienced software developer with strong skills in Python, Java, and cloud computing. Worked on various large-scale projects and proficient in DevOps tools like Docker and Kubernetes.",
    "A creative graphic designer with extensive experience in Adobe Photoshop, Illustrator, and UI/UX design. Skilled in creating impactful visual content for digital platforms.",
    "Data scientist proficient in machine learning, data analysis, and big data tools like Hadoop and Spark. Strong skills in Python, R, and SQL."
]

# Run predictions for the test resumes and print results
for resume in test_resumes:
    predicted_categories, category_probabilities = predict(resume, model, mlb, tokenizer, device)
    print(f"Resume: {resume}")
    print(f"Predicted Job Categories: {predicted_categories}")
    print("Raw Prediction Probabilities:")
    for category, probability in category_probabilities.items():
        print(f"{category}: {probability:.4f}")
    print('-' * 80)
