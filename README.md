# CARAG: Context-Aware Retrieval and Explanation Generation Framework

Welcome to the official repository for CARAG, an end-to-end explainable framework for Automated Fact Verification (AFV) systems. This repository includes the source code and dataset used in our research, which introduces CARAG as a novel methodology integrating thematic embeddings and Subset of Interest (SOI) for contextually aligned and transparent fact verification.

---

## Overview

**CARAG enhances AFV systems by:**

- Leveraging thematic embeddings derived from a focused SOI to incorporate local and global perspectives.
- Integrating evidence retrieval and explanation generation, ensuring alignment with topic-based thematic contexts.
- Providing transparency and explainability through SOI graphs and thematic embeddings.

**Key highlights:**

- **FactVer Dataset:** A novel, explanation-focused dataset specifically curated to enhance thematic alignment and transparency in AFV.
- **Evaluation:** CARAG outperforms standard Retrieval-Augmented Generation (RAG) methods in thematic alignment and explainability.

---

## Features

- **Context-Aware Evidence Retrieval:** Combines thematic embeddings with claim-specific vectors.
- **Explanation Generation Pipeline:** Utilizes Large Language Models (LLMs) in a zero-shot paradigm.
- **Visualization:** Supports thematic SOI graphs for enhanced transparency.

---

## Dataset

- The FactVer dataset, published on Hugging Face @ [FactVer Dataset](https://huggingface.co/datasets/manjuvallayil/factver_master), provides structured evidence relationships and human-generated explanations across multiple topics. It is a key component of CARAG’s evaluation.
- The dataset is also available as a CSV file in this repository (`factver_master.csv`).

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `transformers`.

### Installation

Clone the repository:

```bash
git clone https://github.com/manjuvallayil/factver_dev.git
cd factver_dev
pip install -r requirements.txt
```
---

### 🛠️ Running the Framework

1. **Initialize Parameters**: Modify parameters in `carag_eval.ipynb` as needed:
   - **Dataset:** FactVer (`manjuvallayil/factver_master`)
   - **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`
   - **LLaMA Model:** `meta-llama/Llama-2-7b-chat-hf`
   - **Other parameters:** Similarity threshold (`delta`), weight parameter (`alpha`), and number of documents (`n_docs`) to retrieve.

2. **Run Evaluation Notebook**: Open and run `carag_eval.ipynb` to execute the CARAG pipeline, including:
   - Loading the dataset and models.
   - Generating thematic embeddings and SOI.
   - Evidence retrieval and explanation generation.
   - Visualizing the SOI graph.

3. **Output**:  
   The notebook generates quantitative and qualitative evaluation metrics and visualizations showcasing CARAG’s thematic alignment and explainability.

---


### 📦 Repository Structure

- **datasets/**: Contains the serialized FactVer dataset with textual data and embeddings, ready for retrieval operations.
- **faiss/**: Stores the FAISS index for efficient similarity-based retrieval of evidence embeddings.
- **utils/dataUtils.py**: Provides utilities for loading, grouping, and filtering datasets, including theme-based filtering, embedding extraction, and support for both theme-specific and full-dataset operations in AFV tasks.
- **utils/graphUtils.py**: Includes functions for creating, visualizing, and analyzing graphs that represent relationships between claims and evidences. Supports interactive visualizations of Subset of Interest (SOI) graphs, cluster-level graphs, and interconnection graphs using NetworkX and Plotly.
- **utils/modelUtils.py**: Provides utilities for working with transformer-based models, embedding generation, and unsupervised clustering with Gaussian Mixture Models (GMM).
- **utils/ragUtils.py**: Manages FAISS index creation and evidence retrieval for RAG and CARAG.
- **utils/soi.py**: Contains utilities for identifying the Subset of Interest (SOI) and performing contextual embedding aggregation. Supports both CARAG and CARAG-U methodologies, enabling refinement of evidence based on similarity thresholds and clustering labels.
- This repository also contains generated visualizations (e.g., SOI graphs, clustering graphs) and CSV files created during CARAG execution for analysis and interpretation.
- **README.md**: Documentation for the repository.
- **carag_eval.ipynb**: Main evaluation notebook for the CARAG framework.

---

