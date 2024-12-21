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
