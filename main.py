# main.ipynb

# Import necessary libraries
import pickle
import pandas as pd
from utils.dataUtils import DataUtils
from utils.modelUtils import ModelUtils
from utils.graphUtils import create_and_save_graph, draw_cluster_graph, draw_interconnections

# Parameters
dataset_name = 'manjuvallayil/factver_master'
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
theme = 'Climate'  # Replace with the specific theme you want to analyze
selected_claim_id = 'Claim_36'  # Replace with the specific claim ID you want to analyze

# Initialize DataUtils and ModelUtils
data_utils = DataUtils(dataset_name)
model_utils = ModelUtils(model_name,embedding_model_name)

# Get themed data
themed_data = data_utils.filter_by_theme(theme)

# Check if themed_data is not empty
if themed_data.empty:
    print("No data found for the specified theme. Please check the theme name and try again.")
else:
    # Get embeddings
    all_texts = []
    for index, row in themed_data.iterrows():
        all_texts.append(row['Claim_text'])
        all_texts.extend(row['Evidence_text'])
    embeddings = model_utils.get_embeddings(all_texts)

    # Ensure embeddings are generated
    if embeddings.size == 0:
        print("No embeddings generated. Please check the data and model.")
    else:
        # Cluster embeddings within the selected theme
        labels = model_utils.cluster_embeddings(embeddings)

        # Check if clustering is correct
        unique_labels = set(labels)
        print(f"Unique clusters identified within the theme {theme}: {unique_labels}")
        """
        # Create and save graph
        graph_filepath = 'graph.pkl'
        create_and_save_graph(model_utils, themed_data, labels, graph_filepath)
        
        # Draw cluster graph for the selected theme
        for cluster_id in unique_labels:
            draw_cluster_graph(themed_data, labels, cluster_id=cluster_id, model_utils=model_utils, title=f'{theme} - Cluster Visualization {cluster_id}')
        
        print("Graph visualization completed and saved as HTML files.")
        """
        # Ensure the selected claim is in the identified cluster
        claim_in_cluster = False
        selected_cluster_id = -1  # Variable to store the correct cluster ID

        for index, row in themed_data.iterrows():
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if f"Claim_{unique_id}" == selected_claim_id:
                selected_cluster_id = labels[index]
                claim_in_cluster = True
                break

        if claim_in_cluster:
            # Draw interconnections for the specific claim within the identified cluster
            print(f"The selected claim belongs to cluster {selected_cluster_id}")
            draw_interconnections(themed_data, labels, cluster_id=selected_cluster_id, selected_claim_id=selected_claim_id, model_utils=model_utils, title=f' Thematic and Evidence Interconnections of a claim in {theme} theme')
        else:
            print(f"Selected claim {selected_claim_id} is not part of any identified cluster")