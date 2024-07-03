import pandas as pd
import re
import numpy as np
from datasets import load_dataset

class DataUtils:
    """
    DataUtils handles loading and preprocessing of datasets from Hugging Face Hub.
    """
    def __init__(self, dataset_name, dataset_split='train'):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.factver_data = self.load_data()
        self.grouped_data = self.group_data()

    def load_data(self):
        dataset = load_dataset(self.dataset_name, split=self.dataset_split, trust_remote_code=True)
        data = pd.DataFrame(dataset)
        data['Claim_text'] = data['Claim_text'].astype(str)
        data['Evidence_text'] = data['Evidence_text'].astype(str)
        data.fillna('Missing Data', inplace=True)
        return data

    def group_data(self):
        grouped = self.factver_data.groupby('Claim_topic_id').agg({
            'Claim_text': 'first',
            'Evidence_text': lambda x: list(x),
            'Label': 'first'
        }).reset_index()
        grouped = grouped.sort_values(by='Claim_topic_id', key=lambda x: x.map(self.numeric_sort_key))
        return grouped

    @staticmethod
    def numeric_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def filter_by_theme(self, theme):
        if not hasattr(self, 'grouped_data'):
            self.grouped_data = self.group_data()
        # Extract the theme part from 'Claim_topic_id'
        theme_pattern = f"{theme}_"  # Matches the theme at the start of the 'Claim_topic_id'
        filtered_data = self.grouped_data[self.grouped_data['Claim_topic_id'].str.contains(theme_pattern, case=False)]
        return filtered_data

    def get_embeddings_for_clustering(self, model_utils):
        embeddings = []
        for index, row in self.grouped_data.iterrows():
            if row['Claim_text'].strip():  # Ensuring text is not empty
                claim_embeddings = model_utils.get_embeddings([row['Claim_text']])
                embeddings.extend(claim_embeddings)
            for evidence in row['Evidence_text']:
                if evidence.strip():  # Ensuring text is not empty
                    evidence_embeddings = model_utils.get_embeddings([evidence])
                    embeddings.extend(evidence_embeddings)
        return np.array(embeddings)  # Ensure to return a proper NumPy array