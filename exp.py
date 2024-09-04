# main.py

# Import necessary libraries
import os
import torch
import pandas as pd
from utils.dataUtils import DataUtils
from utils.modelUtils import ModelUtils
from utils.limeUtils import LIMEUtils
from utils.graphUtils import create_and_save_graph, draw_cluster_graph, draw_soi
from utils.soiUtils import SOIUtils
from utils.ragUtils import RAGUtils
from transformers import AutoTokenizer, AutoModelForCausalLM

# Parameters
dataset_name = 'manjuvallayil/factver_master'
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
llama_model_name = 'meta-llama/Llama-2-7b-chat-hf'
theme = 'Climate'
selected_claim_id = 'Claim_36'
similarity_threshold = 0.75
top_k = 2

# Paths for RAGUtils
passages_path = '/home/qsh5523/Documents/factver_dev/dataset'
index_path = '/home/qsh5523/Documents/factver_dev/faiss/index.faiss'

# Initialize LLaMA model
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

def generate_llm_summary(claim, evidences):
    combined_evidence = ' '.join([evidence for evidence in evidences])
    prompt = f"Claim: {claim}\nEvidence: {combined_evidence}\nYou are a fact verification assistant. From the given Claim and its Evidence, determine if the claim is supported by the evidence and generate a concise explanation (two sentences max)."
    
    with torch.no_grad():
        inputs = llama_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = llama_model.generate(inputs['input_ids'], max_new_tokens=200)
    
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Initialize utilities
data_utils = DataUtils(dataset_name)
model_utils = ModelUtils(model_name, embedding_model_name)
lime_utils = LIMEUtils(model_utils)
soi_utils = SOIUtils(model_utils)
rag_utils = RAGUtils(passages_path, index_path, embedding_model_name)

# Load themed data
themed_data = data_utils.filter_by_theme(theme)

# Check if themed data is available
if themed_data.empty:
    print("No data found for the specified theme.")
else:
    # Get embeddings
    all_texts = [row['Claim_text'] for _, row in themed_data.iterrows()]
    for _, row in themed_data.iterrows():
        all_texts.extend(row['Evidence_text'])

    embeddings = model_utils.get_sent_embeddings(all_texts)

    # Cluster embeddings within the selected theme
    labels = model_utils.cluster_embeddings(embeddings)
    unique_labels = set(labels)
    print(f"Unique clusters identified within the theme {theme}: {unique_labels}")
    """
    graph_filepath = 'graph.pkl'
    create_and_save_graph(model_utils, themed_data, graph_filepath)

    # Draw cluster graph for the selected theme
    for cluster_id in unique_labels:
        draw_cluster_graph(themed_data, labels, cluster_id=cluster_id, model_utils=model_utils, title=f'{theme} - Cluster Visualization {cluster_id}')
    """
    # Ensure the selected claim is in the identified cluster
    selected_cluster_id = None
    claim_text = None
    annotated_evidences = None

    for index, row in themed_data.iterrows():
        unique_id = row['Claim_topic_id'].split('_')[-1]
        if f"Claim_{unique_id}" == selected_claim_id:
            selected_cluster_id = labels[index]
            claim_text = row['Claim_text']
            annotated_evidences = row['Evidence_text']
            break

    if selected_cluster_id is not None:
        print(f"The selected claim ({selected_claim_id}) belongs to cluster {selected_cluster_id}")

        # Compute the SOI
        soi = soi_utils.compute_soi(selected_claim_id, themed_data, labels, selected_cluster_id, similarity_threshold)
        soi_evidences = soi['annotated_evidences'] + soi['thematic_cluster_evidences']
        print(f'SOI Evidences: {soi_evidences}')
        draw_soi(soi, similarity_threshold, title=f'SOI Visualization for {selected_claim_id}')
        

        """
        # Clear the GPU cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Generate SOI-based explanation
        soi_explanation = generate_llm_summary(claim_text, [evidence for evidence, _ in soi_evidences])
        print("\nSOI-based Explanation:\n", soi_explanation)

        # Annotated evidences explanation
        if claim_text is not None and annotated_evidences is not None:
            annotated_explanation = generate_llm_summary(claim_text, annotated_evidences)
            print("\nAnnotated Evidences Explanation:\n", annotated_explanation)
        
        # Generate RAG-based explanation
        rag_evidence = rag_utils.retrieve_evidence(claim_text)
        rag_explanation = generate_llm_summary(claim_text, rag_evidence)
        print("\nRAG-based Explanation:\n", rag_explanation)
        """
        
        # Compute aggregated embedding for the SOI evidences
        aggregated_embedding = rag_utils.compute_aggregated_embedding([evidence for evidence, _ in soi_evidences])
        # Retrieve evidence using the aggregated embedding
        agg_evidence = rag_utils.retrieve_evidence(claim_text, aggregated_embedding)
        # Generate explanation using the retrieved evidence from the aggregated context
        agg_context_explanation = generate_llm_summary(claim_text, agg_evidence)
        print("\nSOI's Aggregated Context Explanation:\n", agg_context_explanation)

    else:
        print(f"Selected claim {selected_claim_id} is not part of any identified cluster.")
