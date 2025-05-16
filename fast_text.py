import argparse

import fasttext
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import embedding

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

def load_fasttext_model(path: str):
    return fasttext.load_model(path)


def get_sentence_embedding(text, model):
    words = text.split()
    word_vectors = [model.get_word_vector(word) for word in words if word.strip()]
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.get_dimension())
    

def create_embeddings(df, ft_model):
    embeddings = []
    for feature in clean_text_features:
        df[feature] = df[feature].fillna(' ')
        embeddings += [np.vstack(df[feature].apply(lambda x: get_sentence_embedding(x, ft_model)))]
    return embeddings


def train_both(data_path, ft_path):
    """Train models with both numeric and text features"""
    # data loading and cleaning
    df, targets, text_features, num_features = load_data(data_path)
    df = clean_dataset(df, text_features)
    ft_model = load_fasttext_model(ft_path)
    embeddings = create_embeddings(df, ft_model)
    scores = []
    for n_components in np.arange(5,45,5):
        pca = PCA(n_components=n_components)
        features = []
        for i, embeds in enumerate(embeddings):
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
        scores += [score]
        print(f"test f1: {score}")
        print(f"="*12)
        return scores
    
def train_numeric(data_path, ft_path):
    """Train models with numeric features only."""
    df, targets, text_features, num_features = load_data(data_path)
    df = clean_dataset(df, text_features)
    scores_numeric = []

    num_features_data = df[num_features].values
    data = num_features_data
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
    # print("best params", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    # print(classification_report(y_test, y_pred))
    score = f1_score(y_test, y_pred, average='macro')
    scores_numeric += [score]
    return scores_numeric


def train_text(data_path, ft_path):
    """Train models with text features only"""
    df, targets, text_features, num_features = load_data(data_path)
    df = clean_dataset(df, text_features)
    ft_model = load_fasttext_model(ft_path)
    embeddings = create_embeddings(df, ft_model)
    scores_text = []
    for n_components in np.arange(5,45,5):
        pca = PCA(n_components=n_components)
        features = []
        for i, embeds in enumerate(embeddings):
            features += [pca.fit_transform(embeds)]

        n_features, n_samples, n_dim = np.stack(features).shape
        text_features_data = np.stack(features).transpose(1, 0, 2).reshape(n_samples, n_dim * n_features)

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
                # random_state=state,
                # n_jobs=-1
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
        scores_text += [score]
        print(f"test f1: {score}")
        return scores_text
    

def main(data_path, ft_path, method: str):
    if method == 'numeric':
        return train_numeric(data_path, ft_path)
    elif method == 'text':
        return train_text(data_path, ft_path)
    else:
        return train_both(data_path, ft_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a balanced random forest on tabular and text data.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--ft_path', type=str, required=True, help="Path to the fasttext model.")
    parser.add_argument('--method', type=str, required=True, help="What data to use: `numeric`, `text`, or `both`.")
    args = parser.parse_args()
    scores = main(args.data_path, args.ft_path, args.method)