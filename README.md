# Automated Resume Screening Using Text Embeddings and Machine Learning

This repository contains the source code and data for a bachelor's thesis project aimed at automating resume evaluation using modern natural language processing (NLP) and machine learning techniques.

The project focuses on the classification of Russian-language resumes based on their suitability for analytical roles. The proposed system leverages both structured and unstructured data, including text descriptions from resumes, and applies methods such as TF-IDF, FastText, and BERT for vectorization. It also addresses class imbalance and explores semi-supervised learning with self-training.

## Repository Structure

```
├── bert.py              # BERT vectorization pipeline
├── fast_text.py         # FastText embedding and dimensionality reduction
├── tf_idf.py            # TF-IDF vectorization module
├── setup.py             # Pipeline configuration and model runner
├── utils.py             # Utility functions
├── resumes_all.csv      # Dataset with parsed resumes
├── requirements.txt     # List of required Python libraries
```

## Highlights

- Preprocessing of textual resume fields
- Vectorization using TF-IDF, FastText, and BERT
- Dimensionality reduction via PCA
- Class imbalance handling (undersampling, SMOTE)
- Model training and hyperparameter tuning (Random Forest)
- Evaluation using macro F1-score and other metrics
- Self-training for semi-supervised learning
- Clustering for resume segmentation

## Getting Started

To run the project locally:

```bash
git clone https://github.com/your_username/your_repo.git
cd your_repo
pip install -r requirements.txt
python setup.py
```

Ensure that the `resumes_all.csv` file is in the same directory.

## Author

**Elizaveta Zakharenkova**  
Faculty of Computer Sciences, HSE University  
Bachelor’s Thesis (2025)
