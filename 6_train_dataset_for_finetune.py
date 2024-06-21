
from datasets import load_dataset
# Load the dataset from HuggingFace
fv_dataset = load_dataset('manjuvallayil/factver_master')

# Function to create formatted text for training
def format_data(entry):
    return f"<s>[INST] <<SYS>> You are an Automated fact verification assistant. You are supposed to classify a Claim into one of the following label categories (T for True, F for False and N for Not Enough Info) and you also have to generate an Evidence as an explanation, keep the answer as concise as possible. If a claim does not make any sense, or is not factually coherent, explain why instead of answering something not correct.<</SYS>> Claim: {entry['Claim_text']}. [/INST] Label: {entry['Label']}. Evidence: {entry['Evidence_text']}</s>"


# Apply the formatting function to each row in the dataset
formatted_dataset = fv_dataset.map(lambda x: {'text': format_data(x)}, remove_columns=['Annotation_id', 'Claim_topic_id', 'Evidence_topic_id', 'Claim_text', 'Label', 'Evidence Label', 'Evidence_text', 'Article_topic_id', 'Reason'])

# Example of accessing formatted data
print(formatted_dataset['train']['text'][:1])
formatted_dataset.save_to_disk('factver_dataset')
