"""
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk('/home/qsh5523/Documents/factver/dataset')

# Print some samples from the dataset
print(dataset[:5])
"""

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
#os.environ['HUGGINGFACE_API_TOKEN']="---"
HUGGINGFACE_API_TOKEN = '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'
dataset_name = 'manjuvallayil/factver_master'
page_content_column = "Evidence_text"
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()
print(data[:5])