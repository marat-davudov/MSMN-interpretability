# MSMN Interpretability Demo

An interactive Streamlit demo for visualizing model interpretability of the MSMN (Multiple Synonyms Matching Network) model for automatic ICD coding. This project adapts code from the original [ICD-MSMN implementation](https://github.com/GanjinZero/ICD-MSMN) from the paper "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding" [ACL 2022].

## Features

- 🔍 **Dual Interpretability Methods**: View both attention scores and Integrated Gradients side-by-side
- 🎨 **Color-coded Visualization**: Highlighting for attention, and IG
- 🎯 **Top-K Filtering**: Focus on the most important tokens with configurable top-K selection
- 🏥 **ICD Code Predictions**: Predict and explore multiple ICD codes for medical notes

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- Access to MIMIC-III dataset (for full functionality)

## Dataset

This repository includes **sample data only** for demonstration purposes, taken from the original author's repo. To use the full MIMIC-III dataset:

1. Obtain a license and download the [MIMIC-III dataset](https://mimic.physionet.org/)
2. Follow the preprocessing instructions from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)
3. Place the generated JSON files (`mimic3-50_train.json`, `mimic3-50_dev.json`, `mimic3-50_test.json`) in `sample_data/mimic3/`

## Setup Instructions

### 1. Install Dependencies

```bash
# Install dependencies using Poetry
poetry install
```

### 2. Download Required Files

#### Word Embeddings
Download the word2vec model from LAAT:
```bash
# Create directories
mkdir -p checkpoints/embedding

# Download word2vec model
wget https://github.com/aehrc/LAAT/raw/master/data/embeddings/word2vec_sg0_100.model \
  -O checkpoints/embedding/word2vec_sg0_100.model
```

Or download manually from [here](https://github.com/aehrc/LAAT/blob/master/data/embeddings/word2vec_sg0_100.model) and place in `checkpoints/embedding/`.

#### Model Checkpoint
Download the pre-trained MIMIC-III-50 checkpoint:
```bash
# Create directories
mkdir -p checkpoints/MSMN

# Download from Google Drive (manual download required)
# Visit: https://drive.google.com/file/d/18Ny2R9WLWWa2UpyReaBn-zoSb1Uga9yX/view?usp=sharing
# Save as: checkpoints/MSMN/mimic3-50.pth
```

### 3. Verify Configuration

Check that paths in [constant.py](constant.py) match your setup:
```python
DATA_DIR = "./sample_data/"
EMBEDDING_MODEL_PATH = "./checkpoints/embedding/word2vec_sg0_100.model"
MSMN_MODEL_PATH = "./checkpoints/MSMN/mimic3-50.pth"
```

## Running the Demo

### Start the Streamlit Application

```bash
# Activate the Poetry environment and run Streamlit
poetry run streamlit run demo.py
```

The demo will open in your browser at `http://localhost:8501`.

### Using the Demo

1. **Enter Medical Text**: Paste or type a clinical note in the text area, or select a sample note from the dropdown
2. **Run Inference**: Click "Run Inference" to predict ICD codes and compute interpretability scores
3. **Explore Results**: 
   - Select different predicted ICD codes from the dropdown
   - Adjust the top-K slider to focus on the most important tokens
   - Compare attention scores (left) with Integrated Gradients (right)
4. **Interpret Colors**:
   - **Attention (Green)**: Darker green = higher attention
   - **IG (Red-Green)**: Red = negative attribution, White = neutral, Green = positive attribution

## Configuration

Key parameters in [constant.py](constant.py):
- `IG_STEPS = 30`: Number of steps for Integrated Gradients computation
- `THRESHOLD = 0.5`: Probability threshold for ICD code predictions

## Project Structure

```
.
├── demo.py                 # Main Streamlit application
├── utils.py               # Utility functions (vocab, IG, highlighting)
├── constant.py            # Configuration and paths
├── model/                 # Neural network modules
│   ├── icd_model.py      # Main MSMN model
│   ├── decoder.py        # Attention decoder
│   ├── text_encoder.py   # Text encoding
│   └── ...
├── sample_data/          # Sample MIMIC-III data
└── checkpoints/          # Model weights and embeddings
```

## Technical Details

- **Model Architecture**: LSTM encoder + Multi-head label attention decoder
- **Interpretability**: 
  - Attention weights from the decoder's multi-head attention mechanism
  - Integrated Gradients computed using Captum library
- **Framework**: PyTorch with Streamlit frontend

# Citation
```
@inproceedings{yuan-etal-2022-code,
    title = "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic {ICD} Coding",
    author = "Yuan, Zheng  and
      Tan, Chuanqi  and
      Huang, Songfang",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.91",
    pages = "808--814",
    abstract = "Automatic ICD coding is defined as assigning disease codes to electronic medical records (EMRs).Existing methods usually apply label attention with code representations to match related text snippets.Unlike these works that model the label with the code hierarchy or description, we argue that the code synonyms can provide more comprehensive knowledge based on the observation that the code expressions in EMRs vary from their descriptions in ICD. By aligning codes to concepts in UMLS, we collect synonyms of every code. Then, we propose a multiple synonyms matching network to leverage synonyms for better code representation learning, and finally help the code classification. Experiments on the MIMIC-III dataset show that our proposed method outperforms previous state-of-the-art methods.",
}
```
