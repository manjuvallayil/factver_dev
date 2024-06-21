## this code retrives evidence using RAG and feed it to llama the generates an explanation
## separate function is defined to generate explanation using RAG itself, but not working(returns the query itself)

from transformers import AutoTokenizer, AutoModelForCausalLM, RagTokenizer, RagTokenForGeneration, RagRetriever, RagConfig, RagSequenceForGeneration
import torch
from langchain_community.document_loaders import HuggingFaceDatasetLoader
import os

huggingface_token = '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'
# Set the environment variable to trust remote code
os.environ['TRUST_REMOTE_CODE'] = 'True'

# Initialize components
config = RagConfig.from_pretrained('facebook/rag-token-nq')
config.n_docs = 6
config.passages_path  = '/home/qsh5523/Documents/factver/dataset'
config.index_name = 'custom'  # Not using a Hugging Face index
config.index_path = '/home/qsh5523/Documents/factver/faiss/index.faiss'
config.use_dummy_dataset = False
# Initialize RAG components
tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', config=config)
#rag_model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq', config=config, retriever=retriever)
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', config=config, retriever=retriever)

# Initialize LLaMA
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=huggingface_token)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") #, token=huggingface_token

def evaluate_claim_with_retrieved_evidence(query):
    # Tokenize the query for RAG
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    question_hidden_states = rag_model.question_encoder(input_ids)[0]
    # Convert tensor to numpy array before retrieval if it's on GPU or not
    if question_hidden_states.is_cuda:
        question_hidden_states_np = question_hidden_states.cpu().numpy()
    else:
        question_hidden_states_np = question_hidden_states.detach().numpy()
    # Retrieve documents based on converted numpy array
    try:
        retrieved_doc_embeds, doc_ids, doc_dicts = retriever.retrieve(question_hidden_states_np, n_docs=2)
        retrieved_evidence = doc_dicts[0]['text'][1]
        print(len(retrieved_evidence))
        print("\nRetrieved document:", retrieved_evidence)
    except Exception as e:
        print("An error occurred during retrieval:", str(e))
    # Prepare input for LLaMA using the retrieved evidence
    llama_input = f"Claim: {query}\nEvidence: {retrieved_evidence}\nIs the claim supported by the evidence? Generate an explnation for this, two sentences maximum and keep the answer as concise as possible."
    llama_inputs = llama_tokenizer(llama_input, return_tensors="pt").input_ids
    
    # Generate output from LLaMA
    with torch.no_grad():
        llama_outputs = llama_model.generate(llama_inputs, max_length=200)
    
    # Decode and return the response from LLaMA
    final_response = llama_tokenizer.decode(llama_outputs[0], skip_special_tokens=True)
    return final_response

def evaluate_claim_with_rag(query): ## this funtion returns the query itself not working as intended
    # Tokenize the query for RAG
    inputs = tokenizer(query, return_tensors="pt")
    # Generate outputs from RAG (retrieve and generate)
    with torch.no_grad():
        outputs = rag_model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            use_cache=False,  # Ensures that fresh retrieval is attempted every time
            output_attentions=True  # Helps understand attention mechanism focusing during generation
        )
    # Decode the output from RAG
    #retrieved_evidence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results

# Example usage
query = "Ursula von der Leyen is the European union comission president"

# generate with RAG
#response = evaluate_claim_with_rag(query)
#print("Response from RAG:", response)

# generate with llama
response = evaluate_claim_with_retrieved_evidence(query)
print("Response from LLaMA with Retrieved Evidence:", response)
