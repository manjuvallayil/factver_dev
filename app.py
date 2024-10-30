# app.py

import streamlit as st
import pandas as pd
from utils.dataUtils import DataUtils
from utils.modelUtils import ModelUtils
from utils.soiUtils import SOIUtils
from utils.ragUtils import RAGUtils
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Parameters
dataset_name = 'manjuvallayil/factver_master'
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
llama_model_name = 'meta-llama/Llama-2-7b-chat-hf'
similarity_threshold = 0.75

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
soi_utils = SOIUtils(model_utils)
rag_utils = RAGUtils(passages_path, index_path, embedding_model_name)

# Streamlit UI for theme and claim selection
st.title("Fact Verification Explanation Generator")

# Select theme from sidebar
themes = ["Climate", "Covid", "Electric_Vehicles"]  
selected_theme = st.sidebar.radio("Select a Theme", themes)

# Load themed data
themed_data = data_utils.filter_by_theme(selected_theme)

if themed_data.empty:
    st.write(f"No data found for the selected theme: {selected_theme}.")
else:
    st.write(f"### Claims in {selected_theme} Theme")

    # Combine Claim ID and Claim Text for selection
    claims_df = pd.DataFrame(themed_data[['Claim_topic_id', 'Claim_text']])
    claims_df['Claim ID'] = claims_df['Claim_topic_id'].apply(lambda x: x.split('_')[-1])  # Simplify Claim ID
    claims_df['Claim Display'] = claims_df.apply(lambda row: f"{row['Claim ID']}: {row['Claim_text'][:100]}...", axis=1)

    # Dropdown to select a claim
    selected_claim_display = st.selectbox("Select a Claim to test:", claims_df['Claim Display'])
    
    # Extract selected Claim ID
    selected_claim = claims_df.loc[claims_df['Claim Display'] == selected_claim_display, 'Claim ID'].values[0]
    claim_text = claims_df.loc[claims_df['Claim ID'] == selected_claim, 'Claim_text'].values[0]

    st.write(f"**Selected Claim:** {claim_text}")

    # Generate explanation for the selected claim
    if selected_claim:
        # Generate embeddings for claims
        claim_texts = themed_data['Claim_text'].values.tolist()
        claim_embeddings = model_utils.get_sent_embeddings(claim_texts)

        # Cluster embeddings within the selected theme
        labels = model_utils.cluster_embeddings(claim_embeddings)
        
        # Fix the selection of cluster ID by properly indexing
        filtered_claim_topic_ids = themed_data['Claim_topic_id'].apply(lambda x: x.split('_')[-1])
        selected_cluster_id = labels[filtered_claim_topic_ids == selected_claim].tolist()
        
        if len(selected_cluster_id) == 1:
            selected_cluster_id = selected_cluster_id[0]
        else:
            st.write("Error: Multiple or no matching clusters found for the selected claim.")
            st.stop()  # Stop execution if there's an issue with cluster matching

        # Compute SOI for the selected claim
        soi = soi_utils.compute_soi(f"Claim_{selected_claim}", themed_data, labels, selected_cluster_id, similarity_threshold)
        soi_evidences = soi['annotated_evidences'] + soi['thematic_cluster_evidences']

        # Display Annotated Evidences
        st.write("### Annotated Evidences:")
        for i, (evidence_text, evidence_id) in enumerate(soi['annotated_evidences']):
            st.write(f"{i+1}. {evidence_text}")

        # Compute aggregated embedding for the SOI evidences
        aggregated_embedding = rag_utils.compute_aggregated_embedding([evidence for evidence, _ in soi_evidences])

        # Retrieve evidence using the aggregated embedding
        agg_evidence = rag_utils.retrieve_evidence(claim_text, aggregated_embedding)

        # Generate explanation using the retrieved evidence from the aggregated context
        agg_context_explanation = generate_llm_summary(claim_text, agg_evidence)
        
        st.write(f"### SOI's Aggregated Context Explanation:\n{agg_context_explanation}")