# this code tries to use the embedded dataset and index (which are saved locally by running the first code) and retrive documents, 
# and then to generate, using RAG retriver and RAG model, 
# impression - it generates an empty response
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever, RagConfig, AutoTokenizer, RagSequenceForGeneration
import torch
import os

# Set the environment variable to trust remote code
os.environ['TRUST_REMOTE_CODE'] = 'True'

# Initialize components
config = RagConfig.from_pretrained('facebook/rag-token-nq')
config.n_docs = 6
config.passages_path  = '/home/qsh5523/Documents/factver/dataset'
config.index_name = 'custom'  # Not using a Hugging Face index
config.index_path = '/home/qsh5523/Documents/factver/faiss/index.faiss'
config.use_dummy_dataset = False

tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', config=config)
model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', config=config, retriever=retriever)

# Load dataset
dataset_name = 'manjuvallayil/factver_master'
page_content_column = "Evidence_text"
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
dataset = loader.load()

# Example of processing a single claim with the first document
sample_doc = dataset[0]  # Assume we're using the first document in your dataset
claim = sample_doc.metadata['Claim_text']  # The claim text as the question
print("Query:", claim)

input_text = f"Claim: {claim}\nIs the claim supported by the retrived statements by RAG retriever?"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Use RAG to generate an answer based on the question
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model.generate(input_ids=input_ids)
# Decode the generated tokens to get the answer
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated Answer:", response[0])


