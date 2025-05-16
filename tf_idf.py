#!/usr/bin/env python3

import argparse
import warnings

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from utils import clean_dataset, load_data

warnings.filterwarnings('ignore')

def main(path):
    # data loading and cleaning
    df, targets, text_features, num_features = load_data(path)
    df = clean_dataset(df, text_features)

    # configuration
    only_numeric = False
    state = 44

    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            *[(f'text_{col}', TfidfVectorizer(), col) for col in text_features]
        ]
    )

    num_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features)
        ]
    )

    # pipeline
    balanced_pipeline = ImbPipeline([
        ('preprocessor', num_preprocessor if only_numeric else preprocessor),
        ('classifier', BalancedRandomForestClassifier(
            random_state=state,
            n_jobs=-1
        ))
    ])

    # train-test split
    X = df[num_features] if only_numeric else df
    y = targets['DA'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=40,
        stratify=y
    )

    # cross-validation and hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=state)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_leaf': [2, 5],
        'classifier__min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=balanced_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    # model training and evaluation
    grid_search.fit(X_train, y_train)
    print("best params", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"test f1: {f1_score(y_test, y_pred, average='macro')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a balanced random forest on tabular and text data.")
    parser.add_argument('--path', type=str, required=True, help="Path to the dataset.")
    args = parser.parse_args()
    main(args.path)