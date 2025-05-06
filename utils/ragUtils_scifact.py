
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

        # Initialize sentence transformer model for embedding aggregation
        self.embedding_model = SentenceTransformer(self.embedding_model_name)


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

        


    def create_dataset_and_faiss(self):
        print("Creating dataset and FAISS index for SciFact...")

        # Load SciFact corpus (contains evidence documents)
        corpus_dataset = load_dataset('allenai/scifact', 'corpus', split='train')

        # Format each example to ensure proper structure
        def format_text(example):
            evidence_title = example['title']  # already a string
            id = example['doc_id'] if 'doc_id' in example else ''
            return {'id': id, 'text': evidence_title}
        
        # Apply formatting
        formatted_dataset = corpus_dataset.map(format_text)
        formatted_dataset = formatted_dataset.remove_columns([col for col in formatted_dataset.column_names if col not in {'id', 'text'}])

        # Generate embeddings
        texts = [item['text'] for item in formatted_dataset]
        embeddings = self.encode_texts(texts)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        print('Embeddings Generated for RAG') if embeddings else print('Embeddings Generated NOT for RAG')

        # Add embeddings to dataset and save locally
        formatted_dataset = formatted_dataset.add_column("embeddings", embeddings.tolist())
        formatted_dataset.save_to_disk(self.passages_path)

        # Create FAISS index directory if it doesn't exist
        faiss_dir = os.path.dirname(self.index_path)
        if not os.path.exists(faiss_dir):
            os.makedirs(faiss_dir)

        # Create and save the FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

        print("FAISS index for SciFact created and saved.")

    def encode_texts(self, texts):
        """Encodes the input texts using the embedding model."""
        return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def compute_aggregated_embedding(self, texts):
        """Computes an aggregated embedding by averaging the embeddings of the given texts."""
        embeddings = self.encode_texts(texts)
        aggregated_embedding = np.mean(embeddings, axis=0)
        return aggregated_embedding

    def retrieve_evidence(self, claim, n_docs, aggregated_embedding=None, alpha=0.5, ):
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
            retrieved_doc_embeds, doc_ids, doc_dicts = self.retriever.retrieve(question_hidden_states_np, n_docs)

            # Retrieve evidence texts
            retrieved_evidence = [
                " ".join(doc['text']) if isinstance(doc['text'], list) else doc['text']
                for doc in doc_dicts
            ]

            # Retrieve original doc_ids (these are the custom 'id' fields you formatted)
            retrieved_original_ids = [doc['id'] if 'id' in doc else None for doc in doc_dicts]

            return retrieved_evidence, doc_ids, retrieved_original_ids
        except Exception as e:
            print("An error occurred during retrieval:", str(e))
            return None
    