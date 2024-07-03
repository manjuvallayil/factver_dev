import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import gc
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

class ModelUtils:
    """
    A utility class for loading and utilizing a transformer-based model for sequence classification.
    """

    def __init__(self, model_name: str):
        """
        Initializes the ModelUtils class by setting the model name, and loading the tokenizer and config.
        """
        logging.basicConfig(level=logging.INFO)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        """
        Loads the model specified by the model name into the most suitable device (GPU or CPU).
        """
        gc.collect()
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config).to(device)
        logging.info(f"Classification model loaded on {device.upper()}")
        model.eval()
        return model

    def model_predict(self, texts: list) -> np.ndarray:
        """
        Predicts class probabilities for given texts using the model.
        """
        predictions = []
        device = self.model.device
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions.append(probs)
        return np.array(predictions)

    def get_embeddings(self, texts: list) -> np.ndarray:
        """
        Extracts embeddings from the last hidden state of the model for a list of texts.
        """
        embeddings = []
        if not texts:
            logging.error("No texts provided for embedding extraction.")
            return np.array(embeddings)  # Return empty array if no texts
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.model.device)
                outputs = self.model(**inputs)
                emb = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                if emb.size == 0:
                    logging.warning(f"Failed to generate embeddings for text: {text}")
                else:
                    embeddings.append(emb.flatten())
        return np.vstack(embeddings) if embeddings else np.array([])  # Ensure output is 2D

    def cluster_embeddings(self, embeddings, n_components=3):
        """
        Clusters embeddings using a Gaussian Mixture Model.
        """
        if embeddings.size == 0:
            logging.error("Empty embeddings received for clustering.")
            return np.array([])  # Return empty array if embeddings are empty
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(embeddings)
        labels = gmm.predict(embeddings)
        return labels

    def get_all_embeddings(self, data_utils, theme):
        """
        Retrieves and aggregates embeddings for all claims and their evidences within a specified theme.
        """
        themed_data = data_utils.filter_by_theme(theme)
        all_texts = []
        for index, row in themed_data.iterrows():
            all_texts.append(row['Claim_text'])
            all_texts.extend(row['Evidence_text'])
        return self.get_embeddings(all_texts)