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
    """
    def filter_by_theme(self, theme):
        if not hasattr(self, 'grouped_data'):
            self.grouped_data = self.group_data()
        # Extract the theme part from 'Claim_topic_id'
        theme_pattern = f"{theme}_"  # Matches the theme at the start of the 'Claim_topic_id'
        filtered_data = self.grouped_data[self.grouped_data['Claim_topic_id'].str.contains(theme_pattern, case=False)]
        return filtered_data
    """

    def filter_by_theme(self, claim_id):
        """
        Filters the grouped data based on the Claim_topic_id and extracts the associated theme for filtering.
        """
        if not hasattr(self, 'grouped_data'):
            self.grouped_data = self.group_data()

        # Find the row with the matching Claim_topic_id
        claim_row = None
        for index, row in self.grouped_data.iterrows():
            # Check if the claim_id matches the last part of the Claim_topic_id (e.g., 'Claim_36')
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if unique_id == claim_id.split('_')[-1]:
                claim_row = row
                break  # Stop searching after the first match

        if claim_row is None:
            print(f"No data found for Claim_topic_id: {claim_id}")
            return pd.DataFrame()  # Return an empty DataFrame if no match

        # Extract the theme from the matched Claim_topic_id
        claim_topic_id = claim_row['Claim_topic_id']
        theme = claim_topic_id.split('_')[1]  # Assuming the theme is the second part of the Claim_topic_id
        print(f"\n The selected Claim belongs to the theme: {theme}")

        # Filter the grouped data for all claims with the same theme
        theme_pattern = f"{theme}_"  # Matches the theme part in the 'Claim_topic_id'
        filtered_data = self.grouped_data[self.grouped_data['Claim_topic_id'].str.contains(theme_pattern, case=False)]

        return theme, filtered_data
    
    def themed_data(self,theme):
        theme_pattern = f"{theme}_"
        themed_data = self.grouped_data[self.grouped_data['Claim_topic_id'].str.contains(theme_pattern, case=False)]
        return themed_data
    
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
    
    def get_full_data(self, claim_id):
        """
        Retrieves the full grouped dataset and ensures the selected claim ID exists within the dataset.
        CARAG-U does not filter data by theme, as it works on the entire dataset.

        Args:
            claim_id (str): The ID of the selected claim (e.g., 'Claim_36').

        Returns:
            pd.DataFrame: The full grouped data if the claim ID is valid; otherwise, an empty DataFrame.
        """
        if not hasattr(self, 'grouped_data'):
            self.grouped_data = self.group_data()

        # Validate the presence of the claim in the dataset
        claim_row = None
        for index, row in self.grouped_data.iterrows():
            # Match the claim ID by the last part of 'Claim_topic_id'
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if unique_id == claim_id.split('_')[-1]:
                claim_row = row
                break

        if claim_row is None:
            print(f"No data found for Claim ID: {claim_id}")
            return pd.DataFrame()  # Return an empty DataFrame if no match

        print(f"Claim ID {claim_id} is valid and exists in the dataset.")
        return self.grouped_data