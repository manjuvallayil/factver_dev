from datasets import load_dataset
import pandas as pd
import re

class DataUtils:
    """
    DataUtils handles loading and preprocessing of datasets from Hugging Face Hub.

    Attributes:
        dataset_name (str): Name of the dataset to load from Hugging Face Hub.
        dataset_split (str): The specific split of the dataset to load, e.g., 'train'.
        factver_data (pd.DataFrame): The loaded and preprocessed dataset as a DataFrame.
    """

    def __init__(self, dataset_name, dataset_split='train'):
        """
        Initializes the DataUtils class with the dataset name and split.
        
        Args:
            dataset_name (str): Name of the dataset on Hugging Face Hub.
            dataset_split (str): Split of the dataset to load, default is 'train'.
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.factver_data = self.load_data()

    def load_data(self):
        """
        Load the dataset from Hugging Face Hub and preprocess it.

        Returns:
            pd.DataFrame: The loaded and preprocessed dataset.
        """
        # Load dataset from Hugging Face Hub
        dataset = load_dataset(self.dataset_name, split=self.dataset_split, trust_remote_code=True)
        
        # Convert to pandas DataFrame
        data = pd.DataFrame(dataset)
        
        # Handle text data and missing values
        data['Claim_text'] = data['Claim_text'].astype(str)
        data['Evidence_text'] = data['Evidence_text'].astype(str)
        data.fillna('Missing Data', inplace=True)
        return data

    @staticmethod
    def numeric_sort_key(s):
        """
        Extracts numeric parts for sorting if the string contains numbers.

        Args:
            s (str): The string to be sorted numerically.

        Returns:
            list: List of string and integer parts to enable proper numeric sorting.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    def group_data(self):
        """
        Group evidence texts by 'Claim_topic_id' and sort by 'Claim_topic_id'.

        Returns:
            pd.DataFrame: Grouped data with each claim and its associated evidence texts.
        """
        grouped = self.factver_data.groupby('Claim_topic_id').agg({
            'Claim_text': 'first',  # Use the first occurrence of claim text in each group
            'Evidence_text': lambda x: list(x),  # Collect all evidences related to each claim into a list
            'Label': 'first'  # Use the first label of each group, assuming label consistency within the group
        }).reset_index()

        # Sort by extracting numeric values from IDs using the static method
        grouped = grouped.sort_values(by='Claim_topic_id', key=lambda x: x.map(DataUtils.numeric_sort_key))
        return grouped
    
    def filter_by_theme(self, theme):
        """
        Filter the grouped data to include only those claims related to a specified theme.

        Args:
            theme (str): The theme to filter by, e.g., 'Covid', 'Climate', 'Electric_Vehicles'.

        Returns:
            pd.DataFrame: Filtered data containing only the groups related to the specified theme.
        """
        if not hasattr(self, 'grouped_data'):
            self.grouped_data = self.group_data()
        theme_pattern = f"^Claims_{theme}_"
        return self.grouped_data[self.grouped_data['Claim_topic_id'].str.contains(theme_pattern)]