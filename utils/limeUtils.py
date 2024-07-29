import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

class LIMEUtils:
    def __init__(self, model_utils):
        """ Initializes the LIME utility class with a model utility object.
        Args:
            model_utils (ModelUtils): An instance of ModelUtils that contains methods for model predictions and embeddings.
        """
        self.model_utils = model_utils
        self.explainer = LimeTextExplainer(class_names=["True", "Not Enough Info", "False"])
    
  # def generate_explanation(self, claim, evidences, top_k=2):
        """ Generates a LIME explanation for the top K evidences based on cosine similarity.
        Args:
            claim (str): The claim text.
            evidences (list of str): List of evidence texts.
            top_k (int): The number of top similar evidences to use for generating explanation.
        Returns:
            An explanation object from LIME.
        
        claim_embeddings = self.model_utils.get_embeddings([claim])
        evidence_embeddings = self.model_utils.get_embeddings(evidences)

        # Calculate cosine similarity and select the top K evidences
        similarities = cosine_similarity(claim_embeddings, evidence_embeddings).flatten()
        top_indices = np.argsort(similarities)[-top_k:]  # Indices of the top K similar evidences
        top_evidences = ' '.join([evidences[i] for i in top_indices])

        full_text = f"\n Claim: {claim}. \n Evidence: {top_evidences}"
        print("Text for explanation:", full_text)

        # Generate and display explanations with LIME
        exp = self.explainer.explain_instance(
            full_text,
            lambda x: self.model_utils.model_predict([x])[0],  # Adapter to handle proper indexing
            num_features=10,
            num_samples=100,  # Reduce sample count if needed
            labels=[0, 1, 2]
        )
        return exp
    """
    def generate_explanation(self, claim, evidences, top_k=2, percentile=80):
        claim_embedding = self.model_utils.get_embeddings([claim])
        evidence_embeddings = self.model_utils.get_embeddings(evidences)

        # Calculate semantic similarity using Sentence-Transformers
        similarities = util.pytorch_cos_sim(claim_embedding, evidence_embeddings).squeeze().cpu().numpy()

        # Set dynamic similarity threshold
        threshold = np.percentile(similarities, percentile)

        # Select top K evidences based on combined similarity
        top_indices = np.argsort(similarities)[-top_k:]
        top_evidences = ' '.join([evidences[i] for i in top_indices if similarities[i] > threshold])

        full_text = f"\n Claim: {claim}. \n Evidence: {top_evidences}"
        exp = self.explainer.explain_instance(
            full_text,
            lambda x: self.model_utils.model_predict([x])[0],
            num_features=10,
            num_samples=100,
            labels=[0, 1, 2]
        )
        return top_evidences, exp