from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, AlbertTokenizer, AlbertModel, AlbertModel
from torch.nn import functional as F
from torch import nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Step 1: Data Processing

# Load the dataset
dataset = pd.read_csv('Master.csv')
# List of all unique claims
unique_claims = dataset['Claim_topic_id'].unique()
# Randomly select two unique claims for holding out
held_out_claims = np.random.choice(unique_claims, 4, replace=False)
# Remove the held-out claims from the list of unique claims
unique_claims = [claim for claim in unique_claims if claim not in held_out_claims]
# Split the remaining data into training and testing sets
train_claims, test_claims = train_test_split(unique_claims, test_size=0.2, random_state=42)
# Extract training, testing, and held-out data based on claim IDs
train_data = dataset[dataset['Claim_topic_id'].isin(train_claims)]
test_data = dataset[dataset['Claim_topic_id'].isin(test_claims)]
held_out_data = dataset[dataset['Claim_topic_id'].isin(held_out_claims)]
# Fill NaN values with empty strings for 'Evidence_text' column
train_data.loc[:, 'Evidence_text'] = train_data['Evidence_text'].fillna('').astype(str)
test_data.loc[:, 'Evidence_text'] = test_data['Evidence_text'].fillna('').astype(str)
held_out_data.loc[:, 'Evidence_text'] = held_out_data['Evidence_text'].fillna('').astype(str)
# Aggregate evidence texts by claim ID
train_aggregated_evidence = train_data.groupby('Claim_topic_id').agg({
    'Claim_text': 'first',
    'Label': 'first',
    'Evidence_text': ' '.join
}).reset_index()
test_aggregated_evidence = test_data.groupby('Claim_topic_id').agg({
    'Claim_text': 'first',
    'Label': 'first',
    'Evidence_text': ' '.join
}).reset_index()
held_out_aggregated_evidence = held_out_data.groupby('Claim_topic_id').agg({
    'Claim_text': 'first',
    'Label': 'first',
    'Evidence_text': ' '.join
}).reset_index()

# Add the first instance of Evidence_text as the Reason column for train and test sets
train_aggregated_evidence['Reason'] = train_data.groupby('Claim_topic_id')['Evidence_text'].first().reset_index()['Evidence_text']
test_aggregated_evidence['Reason'] = test_data.groupby('Claim_topic_id')['Evidence_text'].first().reset_index()['Evidence_text']
held_out_aggregated_evidence['Reason'] = held_out_data.groupby('Claim_topic_id')['Evidence_text'].first().reset_index()['Evidence_text']

#print(f"Train-{train_aggregated_evidence.shape},Test-{test_aggregated_evidence.shape},Held Out-{held_out_aggregated_evidence.shape}")
print(train_aggregated_evidence['Evidence_text'].iloc[0])
print('Reason: ', train_aggregated_evidence['Reason'].iloc[0])
#==============================================================================#

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

class Seq2SeqHead(nn.Module):
    def __init__(self, t5_model_name):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.projection = nn.Linear(768, 512)

    def forward(self, encoder_output, decoder_input_ids=None, attention_mask=None):
        # Project ALBERT's output to match T5's expected shape
        projected_output = self.projection(encoder_output)
        return self.t5(decoder_input_ids=decoder_input_ids, encoder_outputs=(projected_output,), attention_mask=attention_mask)

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name, t5_model_name, num_labels):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained(encoder_name)
        self.classification_head = ClassificationHead(hidden_size=768, num_labels=num_labels)
        self.seq2seq_head = Seq2SeqHead(t5_model_name)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None):
        # Get shared encoder output
        shared_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Use the first token's (CLS token) representation as the pooled output
        pooled_output = shared_output[:, 0]

        # Get label logits from the classification head
        label_logits = self.classification_head(pooled_output)

        # Get sequence-to-sequence output
        seq2seq_output = self.seq2seq_head(encoder_output=shared_output,
                                            decoder_input_ids=decoder_input_ids,
                                            attention_mask=decoder_attention_mask)

        return label_logits, seq2seq_output.logits




def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and freeze ALBERT's layers
model = MultiTaskModel('tals/albert-base-vitaminc-fever', 't5-small', num_labels=3)
model = model.to(device)  # Move the model to GPU
freeze_parameters(model.encoder)

#==============================================================================#

# Load the tokenizers
albert_tokenizer = AlbertTokenizer.from_pretrained('tals/albert-base-vitaminc-fever')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenize training data for ALBERT
albert_inputs = albert_tokenizer(
    (train_aggregated_evidence['Claim_text'] + " " + train_aggregated_evidence['Evidence_text']).tolist(),
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Tokenize training data for T5's input
t5_texts = ["summarize: " + evidence for evidence in train_aggregated_evidence['Evidence_text']]
t5_inputs = t5_tokenizer(
    t5_texts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Tokenize training data for T5's target
t5_targets = t5_tokenizer(
    train_aggregated_evidence['Reason'].tolist(),
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)



# Convert labels to tensor (assuming they are categorical and stored as strings)
label_to_index = {'T': 0, 'F': 1, 'N': 2}  # Define your label mapping here
labels = torch.tensor(train_aggregated_evidence['Label'].map(label_to_index).tolist()).to(device)

# Create a DataLoader (you can adjust the batch_size as needed)

dataset = TensorDataset(
    albert_inputs['input_ids'],
    albert_inputs['attention_mask'],
    labels,
    t5_inputs['input_ids'],
    t5_inputs['attention_mask'],
    t5_targets['input_ids']
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Hyperparameters
learning_rate = 5e-5
num_epochs = 20
weight_classification = 0.5
weight_seq2seq = 0.5

# Setup loss and optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
classification_loss_fn = CrossEntropyLoss()
seq2seq_loss_fn = nn.CrossEntropyLoss()

#==============================================================================#

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader: # assuming dataloader yields batches of input and output token IDs
        optimizer.zero_grad()

        # Separate out the inputs and outputs
        input_ids, attention_mask, labels, decoder_input_ids, decoder_attention_mask, decoder_target_ids = batch

        # Move tensors to the device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).long()  # Ensure labels are of type LongTensor

        decoder_input_ids = decoder_input_ids.to(device)
        decoder_attention_mask = decoder_attention_mask.to(device)

        label_logits, seq2seq_output = model(input_ids, attention_mask, decoder_input_ids)

        classification_loss = classification_loss_fn(label_logits, labels)
        # Pad decoder_target_ids to have a sequence length of 512
        decoder_target_ids_padded = F.pad(decoder_target_ids, pad=(0, 512 - decoder_target_ids.size(1)))
        decoder_target_ids_padded = decoder_target_ids_padded.to(device)


        # Compute the seq2seq loss
        seq2seq_loss = seq2seq_loss_fn(
            seq2seq_output.contiguous().view(-1, seq2seq_output.size(-1)),
            decoder_target_ids_padded.contiguous().view(-1)
        )

        # Combine the losses
        combined_loss = weight_classification * classification_loss + weight_seq2seq * seq2seq_loss

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()

    # Print average loss for the epoch
    #print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(dataloader)}")
    print(f"Epoch {epoch+1}/{num_epochs} - Combined Loss: {total_loss / len(dataloader)}, Classification Loss: {classification_loss.item()}, Seq2Seq Loss: {seq2seq_loss.item()}")

#==============================================================================#

# Prepare the held-out set
held_out_inputs = albert_tokenizer(held_out_aggregated_evidence['Claim_text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
t5_held_out_texts = t5_tokenizer(["summarize: " + evidence for evidence in held_out_aggregated_evidence['Evidence_text']], return_tensors='pt', padding=True, truncation=True, max_length=512)

# Move data to the appropriate device
held_out_inputs = {key: val.to(device) for key, val in held_out_inputs.items()}
t5_held_out_texts = {key: val.to(device) for key, val in t5_held_out_texts.items()}

# Ensure the model is in evaluation mode
model.eval()

# Disable gradient calculations
with torch.no_grad():
    # Forward pass to get predictions
    label_logits, seq2seq_logits = model(input_ids=held_out_inputs['input_ids'],
                                     attention_mask=held_out_inputs['attention_mask'],
                                     decoder_input_ids=t5_held_out_texts['input_ids'])

    # Get the predicted labels
    label_preds = torch.argmax(label_logits, dim=1).cpu().numpy()

    # Decode the seq2seq logits to get the generated text
    generated_texts = [t5_tokenizer.decode(g, skip_special_tokens=True) for g in seq2seq_logits.argmax(dim=2)]

# Map the predicted integers back to their respective labels
index_to_label = {v: k for k, v in label_to_index.items()}
label_predictions = [index_to_label[i] for i in label_preds]

# Save the results to a CSV file
results_df = pd.DataFrame({
    'Claim_text': held_out_aggregated_evidence['Claim_text'],
    'True_Label': held_out_aggregated_evidence['Label'],
    'Predicted_Label': label_predictions,
    'Generated_Text': generated_texts
})

results_df.to_csv('held_out_predictions.csv', index=False)
