# utils/soi.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SOIUtils:
    """
    SOIUtils handles the identification of the Subset of Interest (SOI) and contextual embedding
    aggregation for enhanced explainability in AFV systems.
    """

    def __init__(self, model_utils):
        """
        Initializes the SOIUtils class with a model utility object.
        :param model_utils: An instance of ModelUtils that contains methods for model predictions and embeddings.
        """
        self.model_utils = model_utils

    def extract_claim_and_related_data(self, selected_claim_id, themed_data, labels, selected_cluster_id):
        """
        Extracts the selected claim text and its annotated evidences, plus other claims and their thematic_cluster_evidences in the same cluster.
        
        :param selected_claim_id: The unique ID of the selected claim.
        :param themed_data: DataFrame containing the themed data.
        :param labels: Array of cluster labels for each row in the themed_data.
        :param selected_cluster_id: The cluster ID to which the selected claim belongs.
        :return: A dictionary containing the selected claim text, its annotated evidences, other claims, and their thematic_cluster_evidences.
        """
        selected_claim_text = None
        annotated_evidences = []
        related_claims = []
        thematic_cluster_evidences = []

        for index, row in themed_data.iterrows():
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if f"Claim_{unique_id}" == selected_claim_id:
                selected_claim_text = row['Claim_text']
                annotated_evidences = row['Evidence_text'][:6]  # Get the first 6 evidences related to the claim

            elif labels[index] == selected_cluster_id:
                related_claim_id = f"Claim_{unique_id}"
                related_claim_text = row['Claim_text']
                related_claims.append((related_claim_id, related_claim_text))
                
                for i, evidence_text in enumerate(row['Evidence_text']):
                    evidence_id = f"Evidence_{unique_id}_{i}"
                    thematic_cluster_evidences.append((evidence_text, evidence_id))

        if selected_claim_text is None:
            raise ValueError(f"Claim ID {selected_claim_id} not found in the provided data.")

        return {
            'selected_claim_text': selected_claim_text,
            'annotated_evidences': [(evidence_text, f"Evidence_{selected_claim_id.split('_')[-1]}_{i}") for i, evidence_text in enumerate(annotated_evidences)],
            'related_claims': related_claims,
            'thematic_cluster_evidences': thematic_cluster_evidences
        }

    def compute_soi(self, selected_claim_id, themed_data, labels, selected_cluster_id, similarity_threshold):
        """
        Computes the Subset of Interest (SOI) for a given claim based on a similarity threshold.
        
        :param selected_claim_id: The unique ID of the selected claim.
        :param themed_data: DataFrame containing the themed data.
        :param labels: Array of cluster labels for each row in the themed_data.
        :param selected_cluster_id: The cluster ID to which the selected claim belongs.
        :param similarity_threshold: Threshold to consider evidence or claim relevant.
        :return: A dictionary containing relevant evidences, related claims, their IDs, and their similarities.
        """
        # Extract the selected claim text and related data
        data = self.extract_claim_and_related_data(selected_claim_id, themed_data, labels, selected_cluster_id)

        selected_claim_text = data['selected_claim_text']
        annotated_evidences = data['annotated_evidences']
        related_claims = data['related_claims']
        thematic_cluster_evidences = data['thematic_cluster_evidences']

        # Get embeddings for the selected claim
        selected_claim_embeddings = self.model_utils.get_embeddings([selected_claim_text])[0]

        soi = {
            'claim_id': selected_claim_id,
            'claim': selected_claim_text,
            'annotated_evidences': [],
            'related_claims': [],
            'thematic_cluster_evidences': [],
            'similarities': []
        }

        # Check relevance of annotated evidences
        for evidence_text, evidence_id in annotated_evidences:
            evidence_embeddings = self.model_utils.get_embeddings([evidence_text])[0]
            similarity = cosine_similarity(selected_claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
            if similarity > similarity_threshold:
                soi['annotated_evidences'].append((evidence_text, evidence_id))
                soi['similarities'].append((selected_claim_text, evidence_text, similarity))

        # Check relevance of related claims
        for related_claim_id, claim_text in related_claims:
            claim_embeddings = self.model_utils.get_embeddings([claim_text])[0]
            similarity = cosine_similarity(selected_claim_embeddings.reshape(1, -1), claim_embeddings.reshape(1, -1))[0][0]
            if similarity > similarity_threshold:
                soi['related_claims'].append((claim_text, related_claim_id))
                soi['similarities'].append((selected_claim_text, claim_text, similarity))
        
        # Check relevance of thematic cluster evidences
        for evidence_text, evidence_id in thematic_cluster_evidences:
            evidence_embeddings = self.model_utils.get_embeddings([evidence_text])[0]
            similarity = cosine_similarity(selected_claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
            if similarity > similarity_threshold:
                soi['thematic_cluster_evidences'].append((evidence_text, evidence_id))
                soi['similarities'].append((selected_claim_text, evidence_text, similarity))

        return soi

    def calculate_aggregate_embedding(self, soi):
        """
        Calculates the aggregate embedding for the entire SOI.
        
        :param soi: The dictionary containing the SOI details.
        :return: The aggregate embedding for the SOI.
        """
        all_texts = [soi['claim']] + [text for text, _ in soi['annotated_evidences']] + \
                    [text for text, _ in soi['related_claims']] + [text for text, _ in soi['thematic_cluster_evidences']]
        
        embeddings = self.model_utils.get_embeddings(all_texts)
        aggregate_embedding = np.mean(embeddings, axis=0)
        
        return aggregate_embedding