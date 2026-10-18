import hashlib
import json
import os
from pathlib import Path

import catboost
import mlflow
import mlflow.catboost
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    #Допом будем логировать classification_report
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path(os.getenv("DATA_PATH", 'datasets/bank_additional_pre.csv'))
MODEL_NAME = os.getenv("MODEL_NAME", 'bank_model')
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", 'bank_experiment')

ITERATIONS = int(os.getenv("ITERATIONS", "1000"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.05"))
DEPTH = int(os.getenv("DEPTH", "6"))
LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", 'Logloss')
EVAL_METRIC = os.getenv("EVAL_METRIC", 'PRAUC')
AUTO_CLASS_WEIGHTS = os.getenv("AUTO_CLASS_WEIGHTS", 'Balanced')

THRESHOLD = float(os.getenv("THRESHOLD", "0.74"))
MIN_GAIN = float(os.getenv("MIN_GAIN", "0.003"))

TARGET = 'y'
NUMERIC = ['age', 'campaign', 'emp.var.rate', 'cons.conf.idx', 'euribor3m', 'nr.employed']
CATEGORICAL = ['job', 'education', 'month', 'day_of_week']
NUMERIC_MEDIANS = {'age': 38.0, 'campaign': 2.0, 'emp.var.rate': 1.1, 
    'cons.conf.idx': -41.8, 'euribor3m': 4.857, 'nr.employed': 5191.0}
SEED = 42
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.compose._column_transformer._RemainderColsList"]


def load_and_validate(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep = ';')
    need_cols = NUMERIC + CATEGORICAL + [TARGET]
    missing = set(need_cols) - set(df.columns)
    if missing:
        raise ValueError(f"в данных нет колонок: {sorted(missing)}")
    if len(df) < 1000:
        raise ValueError(f"слишком мало строк: {len(df)}")
    if not set(df[TARGET].unique()) <= {"yes", "no"}:
        raise ValueError(f"неожиданные значения таргета: {df[TARGET].unique()[:5]}")
    df = df[need_cols].copy()
    df[TARGET] = (
        df[TARGET]
        .map({"no": 0, "yes": 1})
        .astype("int8")
    )

    for column in CATEGORICAL:
        df[column] = df[column].fillna("__MISSING__").astype(str)

    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(NUMERIC_MEDIANS[column])
    return df

def build_model(
        iterations: int,
        learning_rate: float,
        depth: int, 
        loss_function: str,
        eval_metric: str,
        auto_class_weights: str) -> CatBoostClassifier:

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        eval_metric=eval_metric,
        auto_class_weights=auto_class_weights,
        random_seed=SEED,
        verbose=100,
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        task_type="CPU",
        allow_writing_files=False,
    )

    return model

def champion_pr_auc(client: MlflowClient) -> tuple[str | None, float | None]:
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None
    return mv.version, client.get_run(mv.run_id).data.metrics.get("pr_auc")

def main() -> dict:
    df = load_and_validate(DATA_PATH)
    features = NUMERIC + CATEGORICAL
    target = TARGET
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=SEED, stratify=df[target])
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=SEED, stratify=y_train_val)
    model = build_model(ITERATIONS, LEARNING_RATE, DEPTH, 
            LOSS_FUNCTION, EVAL_METRIC, AUTO_CLASS_WEIGHTS)
    model.fit(
        x_train, 
        y_train, 
        cat_features=CATEGORICAL, 
        eval_set=[(x_val, y_val)], 
        early_stopping_rounds=100
        )

    proba = model.predict_proba(x_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))

    y_pred = (proba >= THRESHOLD).astype(int)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(
        precision_score(y_test, y_pred, zero_division=0)
    )
    recall = float(
        recall_score(y_test, y_pred, zero_division=0)
    )
    f1 = float(
        f1_score(y_test, y_pred, zero_division=0)
    )

    ##Логирование classification_report в формате deteframe
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )

    report_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            report_rows.append({
                "label": label,
                "precision": values.get("precision", 0),
                "recall": values.get("recall", 0),
                "f1-score": values.get("f1-score", 0),
                "support": values.get("support", 0)
            })
    report_df = pd.DataFrame(report_rows)


    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    with mlflow.start_run() as run:
        metadata = {"feature_names": features, "categorical_features": CATEGORICAL, 
        "numeric_features": NUMERIC, "threshold": round(THRESHOLD, 4), 
        "numeric_medians": NUMERIC_MEDIANS, "categorical_missing_value": "__MISSING__", 
        "n_train": len(x_train), "data_rows": len(df), 
        "catboost_version": catboost.__version__, "sklearn": sklearn.__version__}

        mlflow.log_params({"model": "CatBoostClassifier", "iterations": ITERATIONS, 
            "learning_rate": LEARNING_RATE, "depth": DEPTH, "loss_function": LOSS_FUNCTION, 
            "eval_metric": EVAL_METRIC, "auto_class_weights": AUTO_CLASS_WEIGHTS, 
            "seed": SEED, "data": str(DATA_PATH)
            })

        mlflow.log_metrics({"roc_auc": roc_auc, "pr_auc": pr_auc, "threshold": round(THRESHOLD, 4), 
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_table(report_df, artifact_file="classification_report.json")
        mlflow.log_param("data_md5", hashlib.md5(DATA_PATH.read_bytes()).hexdigest())

        info = mlflow.catboost.log_model(
            model, 
            name = 'bank_model', 
            registered_model_name=MODEL_NAME,
            input_example=x_test.head(5), metadata=metadata
        )
        version = info.registered_model_version

    old_version, old_pr_auc = champion_pr_auc(client)
    promoted = old_pr_auc is None or pr_auc > old_pr_auc + MIN_GAIN
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    
    result = {"run_id": run.info.run_id, "version": version, "pr_auc": pr_auc, 
        "champion_before": old_version, "champion_pr_auc_before": old_pr_auc, "promoted": promoted}
    
    print(json.dumps(result, ensure_ascii=False))
    xcom = Path("/airflow/xcom")
    if xcom.is_dir():
        (xcom / "return.json").write_text(json.dumps(result))
    return result

if __name__ == "__main__":
    main()




        

