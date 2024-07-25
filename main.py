# main.ipynb

# Import necessary libraries
import pickle
import pandas as pd
from utils.dataUtils import DataUtils
from utils.modelUtils import ModelUtils
from utils.limeUtils import LIMEUtils
from utils.graphUtils import create_and_save_graph, draw_cluster_graph, draw_interconnections

# Parameters
dataset_name = 'manjuvallayil/factver_master'
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
theme = 'Climate'  # Replace with the specific theme you want to analyze
selected_claim_id = 'Claim_5'  # Replace with the specific claim ID you want to analyze

# Initialize DataUtils, ModelUtils, and LIMEUtils
data_utils = DataUtils(dataset_name)
model_utils = ModelUtils(model_name, embedding_model_name)
lime_utils = LIMEUtils(model_utils)

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
    embeddings = model_utils.get_sent_embeddings(all_texts)

    # Ensure embeddings are generated
    if embeddings.size == 0:
        print("No embeddings generated. Please check the data and model.")
    else:
        # Cluster embeddings within the selected theme
        labels = model_utils.cluster_embeddings(embeddings)

        # Check if clustering is correct
        unique_labels = set(labels)
        print(f"Unique clusters identified within the theme {theme}: {unique_labels}")

        # Print the number of samples in each cluster
        cluster_sizes = {label: list(labels).count(label) for label in unique_labels}
        for cluster_id, size in cluster_sizes.items():
            print(f"Cluster {cluster_id} has {size} samples.")
        
        # Create and save graph
        graph_filepath = 'graph.pkl'
        create_and_save_graph(model_utils, themed_data, graph_filepath)
        
        # Draw cluster graph for the selected theme
        for cluster_id in unique_labels:
            draw_cluster_graph(themed_data, labels, cluster_id=cluster_id, model_utils=model_utils, title=f'{theme} - Cluster Visualization {cluster_id}')
        
        print("Graph visualization completed and saved as HTML files.")
        
        
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
            print(f"The selected claim belongs to cluster {selected_cluster_id}")
            # Draw interconnections for the specific claim within the identified cluster
            draw_interconnections(themed_data, labels, cluster_id=selected_cluster_id, selected_claim_id=selected_claim_id, model_utils=model_utils, title=f'Thematic and Evidence Interconnections of a claim in {theme} theme')

            # Generate LIME explanation for the baseline
            selected_claim_row = themed_data[themed_data['Claim_topic_id'].str.contains(f'_{selected_claim_id.split("_")[-1]}')]
            if selected_claim_row.empty:
                print(f"No claim found with ID {selected_claim_id} in the thematic subset.")
            else:
                claim = selected_claim_row['Claim_text'].values[0]
                annotated_evidences = selected_claim_row['Evidence_text'].values[0]

                print("Baseline Explanation:")
                baseline_exp = lime_utils.generate_explanation(claim, annotated_evidences, top_k=2)
                baseline_exp.show_in_notebook(text=True)
                print(baseline_exp.as_list())

                # Generate LIME explanation for the thematic cluster
                thematic_cluster_evidences = []
                for idx, row in themed_data.iterrows():
                    if labels[idx] == selected_cluster_id:
                        thematic_cluster_evidences.extend(row['Evidence_text'])

                print("Cluster Explanation:")
                cluster_exp = lime_utils.generate_explanation(claim, thematic_cluster_evidences, top_k=2)
                cluster_exp.show_in_notebook(text=True)
                print(cluster_exp.as_list())

                # Optionally, save the explanations for further analysis
                baseline_exp.save_to_file('baseline_exp.html')
                cluster_exp.save_to_file('cluster_exp.html')

        else:
            print(f"Selected claim {selected_claim_id} is not part of any identified cluster.")