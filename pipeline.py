import pandas as pd
import joblib

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def preprocess_data(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )

    X = preprocessor.fit_transform(X)

    target_encoder = None

    if y.dtype == "object":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y)

    return X, y, preprocessor, target_encoder


def train_models(X_train, y_train):

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "KNN":
            KNeighborsClassifier(n_neighbors=5),

        "Naive Bayes":
            GaussianNB(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models

def evaluate_models(models, X_test, y_test):

    results = []
    reports = {}

    for name, model in models.items():

        y_pred = model.predict(X_test)

        # Automatically identify the positive class
        labels = sorted(list(set(y_test)))

        if len(labels) == 2:
            positive_label = labels[1]
        else:
            positive_label = None

        # Calculate metrics
        if positive_label is not None:

            precision = precision_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                pos_label=positive_label,
                zero_division=0
            )

        else:

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        # AUC
        if hasattr(model, "predict_proba"):

            y_prob = model.predict_proba(X_test)

            if len(labels) == 2:
                y_prob_positive = y_prob[:, 1]

                try:
                    row["AUC"] = roc_auc_score(
                        y_test,
                        y_prob_positive
                    )
                except ValueError:
                    row["AUC"] = None

            else:
                try:
                    row["AUC"] = roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr",
                        average="weighted"
                    )
                except ValueError:
                    row["AUC"] = None

        else:
            row["AUC"] = None

        results.append(row)

        reports[name] = {
            "confusion_matrix":
                confusion_matrix(y_test, y_pred),

            "classification_report":
                classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )
        }

    results_df = pd.DataFrame(results)

    return results_df, reports

def run_pipeline(df, target_column):

    X, y, preprocessor, target_encoder = preprocess_data(
        df,
        target_column
    )

    print(type(y))
    print(pd.Series(y).unique()[:5])

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    models = train_models(
        X_train,
        y_train
    )

    results_df, reports = evaluate_models(
        models,
        X_test,
        y_test
    )


    os.makedirs("saved_models", exist_ok=True)

    for name, model in models.items():

        filename = name.lower().replace(" ", "_") + ".pkl"

        joblib.dump(
        model,
        f"saved_models/{filename}"
        )

    return (
        models,
        results_df,
        reports,
        X_test,
        y_test,
        preprocessor,
        target_encoder
    )

