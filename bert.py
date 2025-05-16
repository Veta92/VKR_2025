import argparse

import numpy as np
import torch
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from utils import clean_dataset, load_data

clean_text_features = ['Название',
 'Коммандировка',
 'Опыт',
 'О себе',
 'Образование',
 'Интересы',
 'Навыки',
 'Уровень образование',
 'Работа 1',
 'Работа 2',
]

def load_model(device: str = 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    tokenizer.model_max_length = 512
    bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(device)
    return bert, tokenizer

@torch.no_grad()
def get_bert_embedding(text, tokenizer, bert):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
    hs = bert(tokens).last_hidden_state # [batch_size, seq_len, hidden_dim]
    embeds = hs[:, 0, :]
    return embeds.cpu().numpy()

def create_embeddings(df, clean_text_features, tokenizer, bert):
    batch_size = 109
    bert_embeddings = []
    for feature in (clean_text_features):
        df[feature] = df[feature].fillna(' ')
        feature_embeds = []
        for batch_idx in (range(25)):
            texts = df[feature].iloc[batch_idx*batch_size:(batch_idx+1)*batch_size].tolist()
            feature_embeds += [get_bert_embedding(texts, tokenizer, bert)]
        bert_embeddings += [np.array(feature_embeds).reshape(-1, 768)]
    return bert_embeddings


def train_both(data_path, device: str = 'cpu'):
    df, targets, text_features, num_features = load_data(data_path)
    df = clean_dataset(df, text_features)
    bert, tokenizer = load_model(device=device)
    bert_embeddings = create_embeddings(df, clean_text_features, tokenizer, bert)
    bert_scores = []
    for n_components in np.arange(5,45,5):
        pca = PCA(n_components=n_components)
        features = []
        for i, embeds in enumerate(bert_embeddings):
            features += [pca.fit_transform(embeds)]

        n_features, n_samples, n_dim = np.stack(features).shape
        text_features_data = np.stack(features).transpose(1, 0, 2).reshape(n_samples, n_dim * n_features)

        num_features_data = df[num_features].values
        data = np.concatenate([text_features_data, num_features_data], axis=1)
        X = data
        y = targets['DA'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.4,
            random_state=40,
            stratify=y
        )

        state = 44
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=state)

        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [2, 5],
            'min_samples_split': [2, 5],
            'random_state': [42,43,44,45]
        }

        model = BalancedRandomForestClassifier(
            )
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print("best params", grid_search.best_params_)

        y_pred = grid_search.predict(X_test)
        print(classification_report(y_test, y_pred))
        score = f1_score(y_test, y_pred, average='macro')
        bert_scores += [score]
        print(f"test f1: {score}")
        return bert_scores
    

def train_text(data_path, device: str = 'cpu'):
    df, targets, text_features, num_features = load_data(data_path)
    df = clean_dataset(df, text_features)
    bert, tokenizer = load_model(device=device)
    bert_embeddings = create_embeddings(df, clean_text_features, tokenizer, bert)
    bert_scores_text = []
    for n_components in np.arange(5,45,5):
        pca = PCA(n_components=n_components)
        features = []
        for i, embeds in enumerate(bert_embeddings):
            features += [pca.fit_transform(embeds)]

        n_features, n_samples, n_dim = np.stack(features).shape
        text_features_data = np.stack(features).transpose(1, 0, 2).reshape(n_samples, n_dim * n_features)

        num_features_data = df[num_features].values
        # data = np.concatenate([text_features_data, num_features_data], axis=1)
        data = text_features_data
        X = data
        y = targets['DA'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.4,
            random_state=40,
            stratify=y
        )

        state = 44
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=state)

        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [2, 5],
            'min_samples_split': [2, 5],
            'random_state': [42,43,44,45]
        }

        model = BalancedRandomForestClassifier(
            )
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print("best params", grid_search.best_params_)

        y_pred = grid_search.predict(X_test)
        print(classification_report(y_test, y_pred))
        score = f1_score(y_test, y_pred, average='macro')
        bert_scores_text += [score]
        print(f"test f1: {score}")
    return bert_scores_text