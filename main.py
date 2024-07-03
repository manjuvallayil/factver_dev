
# Import necessary libraries
import pickle
import pandas as pd
from utils.dataUtils import DataUtils
from utils.modelUtils import ModelUtils
from utils.graphUtils import create_and_save_graph, draw_cluster_graph

# Parameters
dataset_name = 'manjuvallayil/factver_master'
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
theme = 'Electric_Vehicles'  # Replace with any valid theme keyword

# Initialize DataUtils and ModelUtils
data_utils = DataUtils(dataset_name)
model_utils = ModelUtils(model_name)

# List available themes (extracted theme parts)
available_themes = data_utils.grouped_data['Claim_topic_id'].apply(lambda x: x.split('_')[1]).unique()
print("Available themes:", available_themes)

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
        # Cluster embeddings
        labels = model_utils.cluster_embeddings(embeddings)

        # Check if clustering is correct
        unique_labels = set(labels)
        print(f"Unique clusters identified: {unique_labels}")

        # Create and save graph
        graph_filepath = 'graph.pkl'
        create_and_save_graph(model_utils, themed_data, labels, graph_filepath)

        # Draw cluster graph
        for cluster_id in unique_labels:
            draw_cluster_graph(themed_data, labels, cluster_id=cluster_id, model_utils=model_utils, title=f'Cluster Visualization {cluster_id}')

        print("Graph visualization completed and saved as HTML files.")