# Resume Screening Automation (Bachelor's Thesis, 2025)

This repository contains the code and data used in the bachelor's thesis project by Elizaveta Zakharenkova (Faculty of Computer Science, HSE University, 2025), focused on automating resume screening for analytical roles using machine learning and natural language processing techniques.

## 📌 Project Description

The goal of the project was to develop a system capable of analyzing resumes and predicting candidate suitability for analytical positions. A complete workflow was implemented:
- Automatic data collection from the hh.ru platform
- Preprocessing and normalization of text and numeric features
- Manual annotation of part of the dataset
- Classification using traditional models (Random Forest, Logistic Regression)
- Semi-supervised learning via self-training
- Resume clustering for deeper candidate segmentation

Both numerical and textual features were used. Texts were embedded using TF-IDF, FastText, and BERT models. Special attention was paid to class imbalance, addressed using SMOTE and balanced ensemble classifiers.

## 📂 Repository Structure

- `setup.py` — experiment runner and configuration entry point
- `tf_idf.py`, `fast_text.py`, `bert.py` — text vectorization scripts
- `utils.py` — helper functions for loading and processing features
- `resumes_all.csv` — raw parsed resume data
- `resumes_features-6.csv` — **final processed dataset** (available in [Releases](https://github.com/Veta92/VKR_2025/releases))
- `requirements.txt` — environment setup

## 📈 Dataset

The cleaned and final dataset used in training and evaluation is available in the [Releases](https://github.com/Veta92/VKR_2025/releases) section:

- **`resumes_features-6.csv`** — final feature set after all preprocessing, combining numeric and textual fields.
  - Includes standard features (age, experience, education)
  - Extracted text fields: About Me, Skills, Experience
  - Suitable for embedding and training

## 📊 Models

- Classifiers: Logistic Regression, Random Forest, BalancedRandomForestClassifier
- Semi-supervised learning: Self-training on unlabeled data
- Clustering: KMeans, DBSCAN
- Evaluation: macro F1-score, confusion matrices, weighted metrics

## 🤖 Text Embedding

- **TF-IDF** — classical sparse embedding baseline
- **FastText** — subword-aware dense embedding, PCA-compressed
- **BERT** — contextual embedding (DeepPavlov/rubert-base-cased)

## 🔗 Deployment & Usage

All experiments were performed locally using Jupyter and Python 3.10. The code is modular and can be reused for similar tasks on new resume datasets.
