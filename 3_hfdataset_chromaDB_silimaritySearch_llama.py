# this code loads hf dataset then created a vector db(chroma using sentence transformers), then do similarity search for similar documents 
# based on the query and retrive them, this is passed to llama2 along with the query
# impression--quality of the retrived documents is poor

## This approach is similar to "retrieve-and-generate" but lacks the direct, real-time interaction between the retriever and generator typical in RAG.
## Traditional RAG uses a DPR retriever built into its architecture, which directly interacts with the transformer generator to augment the generation process with retrieved context. 
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#Load dataset from huggingface
huggingface_token = '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'
dataset_name = 'manjuvallayil/factver_master'
page_content_column = "Evidence_text"
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
dataset = loader.load()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") #, token=huggingface_token
#Data Cleaning Function
def clean_metadata(document):
    for key, value in document.metadata.items():
        if value is None:
            document.metadata[key] = ""  # Replace None with an empty string
    return document

cleaned_data = [clean_metadata(doc) for doc in dataset]

#Document retrieval and vector store setup
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(cleaned_data, embedding_function)

#Create two separate functions for handling different input types
def evaluate_claim_with_retrieved_evidence(query):
    retrieved_results = db.similarity_search_with_score(query)
    if not retrieved_results:
        return "No results found."
    first_result, score = retrieved_results[1]
    input_text = f"Claim: {query}\nEvidence: {first_result.page_content}\nIs the claim supported by the evidence? Generate an explnation for this, two sentences maximum and keep the answer as concise as possible."
    #print(f"INPUT_TEXT_RETRIEVED:", input_text, "\n Score:", score)
    print(score)
    print(first_result.page_content)
    print(first_result.metadata.get('Label'))
    print(first_result.metadata.get('Evidence_topic_id'))
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    return input_ids

def evaluate_claim_with_given_evidence(claim, evidence, label):
    input_text = f"Claim: {claim}\nEvidence: {evidence}\nIs the claim supported by the evidence?"
    print("INPUT_TEXT_GIVEN:", input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#test
# Using the first function with a query
query = "Covid Vaccine Safe for Children"
input_ids = evaluate_claim_with_retrieved_evidence(query)
outputs = model.generate(input_ids, max_length=300) 
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response with retrieved evidence:", response)

# Using the second function with a specific document
#sample_doc = cleaned_data[0]
#print("Response with given evidence:", evaluate_claim_with_given_evidence(sample_doc.metadata['Claim_text'], sample_doc.page_content, sample_doc.metadata['Label']))

