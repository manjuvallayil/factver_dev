# this code loads hf dataset, embed the 'text' colum of it and saves the encoded db locally, 
# faiss index(.faiss) and pickle file(.pkl) are also saved

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import torch
import faiss
import os

# Load the dataset from disk
#fv_dataset = load_from_disk('/home/qsh5523/Documents/factver/dataset')

# Check the dataset structure
#print("Dataset sample:", fv_dataset['text'][:3])  # Print first three samples

# Load your dataset
dataset = load_dataset('manjuvallayil/factver_master', split='train', trust_remote_code=True)
# Function to ensure the text is a string and replace NaNs with empty strings
def format_text(example):
    if example['Evidence_text'] is None:
        example['Evidence_text'] = ''
    else:
        example['Evidence_text'] = str(example['Evidence_text'])
    return example
# Apply the function to the dataset
dataset = dataset.map(format_text)
# Map dataset to include only necessary text and use 'Evidence_topic_id' as 'title'
filtered_dataset = dataset.map(lambda x: {'title': x['Evidence_topic_id'], 'text': x['Evidence_text']})
# Optionally, remove columns you don't need
filtered_dataset = filtered_dataset.remove_columns([col for col in filtered_dataset.column_names if col not in {'title', 'text'}])


# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

def encode_texts(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Ensuring it's numpy array
        all_embeddings.append(embeddings)

    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)  # Stack to create a single numpy array
        print("Embedding complete. Number of embeddings:", len(all_embeddings))
    return all_embeddings


# Extract texts and ensure they are strings
texts = [item['text'] for item in filtered_dataset]

# Generate embeddings and check length
embeddings = encode_texts(texts)
# Check and prepare embeddings for FAISS
if embeddings.size == 0:
    raise ValueError("No embeddings generated. Check the input texts and model outputs.")
embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)  # Ensure the array is contiguous and of type float32

# Add embeddings to the dataset and save
filtered_dataset = filtered_dataset.add_column("embeddings", embeddings.tolist())
filtered_dataset.save_to_disk('/home/qsh5523/Documents/factver/dataset')

# Save as a pickle file for RAG, if necessary
df_pandas = filtered_dataset.to_pandas()
df_pandas.to_pickle('/home/qsh5523/Documents/factver/psgs_w100.tsv.pkl')


# Create and save the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
if not os.path.exists('faiss'):
   os.makedirs('faiss')
faiss.write_index(index, '/home/qsh5523/Documents/factver/faiss/index.faiss')

print("FAISS index created and saved.")
