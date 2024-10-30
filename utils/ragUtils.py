
import os
import torch
import numpy as np
import faiss
from transformers import RagTokenizer, RagRetriever, RagConfig, RagSequenceForGeneration
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

class RAGUtils:
    def __init__(self, passages_path, index_path, embedding_model_name='sentence-transformers/all-mpnet-base-v2'):
        self.passages_path = passages_path
        self.index_path = index_path
        self.embedding_model_name = embedding_model_name

        # Check if the dataset and FAISS index exist, otherwise create them
        if not os.path.exists(self.passages_path) or not os.path.exists(self.index_path):
            self.create_dataset_and_faiss()

        # Initialize RAG components
        self.config = RagConfig.from_pretrained('facebook/rag-token-nq')
        self.config.n_docs = 6
        self.config.passages_path = self.passages_path
        self.config.index_name = 'custom'
        self.config.index_path = self.index_path
        self.config.use_dummy_dataset = False

        self.tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
        self.retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', config=self.config)
        self.rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', config=self.config, retriever=self.retriever)

        # Initialize sentence transformer model for embedding aggregation
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def create_dataset_and_faiss(self):
        print("Creating dataset and FAISS index...")

        # Load your dataset
        dataset = load_dataset('manjuvallayil/factver_master', split='train', trust_remote_code=True)

        # Function to ensure the text is a string and replace NaNs with empty strings
        def format_text(example):
            if example['Evidence_text'] is None:
                example['Evidence_text'] = ''
            else:
                example['Evidence_text'] = str(example['Evidence_text'])
            return example

        dataset = dataset.map(format_text)

        # Map dataset to include only necessary text and use 'Evidence_topic_id' as 'title'
        filtered_dataset = dataset.map(lambda x: {'title': x['Evidence_topic_id'], 'text': x['Evidence_text']})
        filtered_dataset = filtered_dataset.remove_columns([col for col in filtered_dataset.column_names if col not in {'title', 'text'}])

        # Generate embeddings
        texts = [item['text'] for item in filtered_dataset]
        embeddings = self.encode_texts(texts)

        # Add embeddings to the dataset and save
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        filtered_dataset = filtered_dataset.add_column("embeddings", embeddings.tolist())
        filtered_dataset.save_to_disk(self.passages_path)

        # Create FAISS index directory if it doesn't exist
        faiss_dir = os.path.dirname(self.index_path)
        if not os.path.exists(faiss_dir):
            os.makedirs(faiss_dir)

        # Create and save the FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

        print("FAISS index created and saved.")

    def encode_texts(self, texts):
        """Encodes the input texts using the embedding model."""
        return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def compute_aggregated_embedding(self, texts):
        """Computes an aggregated embedding by averaging the embeddings of the given texts."""
        embeddings = self.encode_texts(texts)
        aggregated_embedding = np.mean(embeddings, axis=0)
        return aggregated_embedding

    def retrieve_evidence(self, claim, aggregated_embedding=None, alpha=0.5):
        # Tokenize the claim for RAG
        input_ids = self.tokenizer(claim, return_tensors="pt").input_ids
        question_hidden_states = self.rag_model.question_encoder(input_ids)[0]

        # If aggregated_embedding is provided, average with claim embedding
        if aggregated_embedding is not None:
            claim_embedding = question_hidden_states.detach().cpu().numpy()  # Detach and convert to numpy
            
            # Ensure both embeddings have the same shape
            if claim_embedding.shape[0] == 1:
                claim_embedding = np.squeeze(claim_embedding, axis=0)  # Remove the extra dimension from claim_embedding

            # Now combine the embeddings
            #combined_embedding = np.mean([claim_embedding, aggregated_embedding], axis=0)
            # Ensure both embeddings have the same shape
            assert claim_embedding.shape == aggregated_embedding.shape
            # Combine the embeddings using the weighted average
            combined_embedding = alpha * claim_embedding + (1 - alpha) * aggregated_embedding
            question_hidden_states_np = np.expand_dims(combined_embedding, axis=0)  # Ensure correct shape for retrieval
        else:
            question_hidden_states_np = question_hidden_states.detach().cpu().numpy()

        # Retrieve documents based on the claim or combined embedding
        try:
            retrieved_doc_embeds, doc_ids, doc_dicts = self.retriever.retrieve(question_hidden_states_np, n_docs=6)
            retrieved_evidence = [" ".join(doc['text']) if isinstance(doc['text'], list) else doc['text'] for doc in doc_dicts]
            #retrieved_evidence = [doc['text'] for doc in doc_dicts]
            #retrieved_evidence = [doc['text'][0] for doc in doc_dicts]
            #print("\nRAG Retrieved documents:", retrieved_evidence)
            return retrieved_evidence
        except Exception as e:
            print("An error occurred during retrieval:", str(e))
            return None
    # New method to load dataset vectors from the FAISS index
    def load_dataset_vectors(self):
        """Loads the dataset vectors from the FAISS index."""
        # Read the FAISS index
        index = faiss.read_index(self.index_path)

        # Check if the index contains vectors
        if index.ntotal == 0:
            raise ValueError("The FAISS index is empty. No vectors found.")

        # Retrieve and return all the vectors in the index
        try:
            vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
            return vectors
        except Exception as e:
            raise RuntimeError(f"Error loading dataset vectors: {str(e)}")