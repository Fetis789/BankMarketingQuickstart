import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

RAW_PATH = Path(os.getenv("DATA_PATH", 'datasets/bank_additional_pre.csv'))
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", 'bank_experiment')

# Правила для каждой строки
ROWS = pa.DataFrameSchema(
    {
        "age": pa.Column(int, pa.Check.in_range(12, 100)),
        "campaign": pa.Column(int, pa.Check.in_range(0, 100)),
        "emp.var.rate": pa.Column(float, pa.Check.in_range(-10, 10)),
        "cons.conf.idx": pa.Column(float, pa.Check.in_range(-60, 0)),
        "euribor3m": pa.Column(float, pa.Check.in_range(0, 10)),
        "nr.employed": pa.Column(float, pa.Check.in_range(3000, 7000)),
        "job": pa.Column(str, pa.Check.isin(["admin.", "blue-collar", "entrepreneur",
                                            "housemaid", "management", "retired", 
                                            "self-employed", "services", "student", 
                                            "technician", "unemployed", "unknown", 
                                            "wild.card"])),
        "education": pa.Column(str, pa.Check.isin(["basic.4y", "basic.6y", "basic.9y", 
                                                "high.school", "professional.course", 
                                                "university.degree", "unknown", 
                                                "illiterate"])),
        "month": pa.Column(str, pa.Check.isin(["jan", "feb", "mar", 
                                                "apr", "may", "jun", 
                                                "jul", "aug", "sep", 
                                                "oct", "nov", "dec"])),
        "day_of_week": pa.Column(str, pa.Check.isin(["mon", "tue", "wed", "thu", "fri"])),
        "y": pa.Column(str, pa.Check.isin(["yes", "no"])),
    }
)


EXPORT = pa.DataFrameSchema(
    ROWS.columns,
    checks =[
        pa.Check(
                lambda df: len(df) >= 10000, 
                error = "Слишком маленький датасет. Меньше 10 тысяч строк"
                ),
        pa.Check(lambda df: len(df[df['y'] == 'yes']) > 0, error = "Нет положительных случаев"),
        pa.Check(lambda df: len(df[df['y'] == 'no']) > 0, error = "Нет отрицательных случаев"),
        pa.Check(
                lambda df: df["y"].eq("yes").mean() < 0.5, 
                error = 'Слишком много положительных случаев'
                ),
        pa.Check(
                lambda df: df["y"].eq("yes").mean() > 0.01, 
                error = 'Слишком мало положительных случаев'
                ),
    ]
    #strict = True
)

def file_md5(path: Path = RAW_PATH) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep = ';')

def main() -> int:
    import mlflow

    df = load_data(RAW_PATH)
    md5 = file_md5(RAW_PATH)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name = 'validate'):
        mlflow.log_params({"step": "validate", "data": str(RAW_PATH), "data_md5": md5})
        mlflow.log_metrics({"num_rows": len(df)})
        try:
            EXPORT.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            fc = e.failure_cases
            summary = (
                fc.groupby(["column", "check"], dropna=False)
                .size()
                .rename("cases")
                .reset_index()
            )
            with tempfile.TemporaryDirectory() as tmp:
                fc.head(1000).to_csv(Path(tmp) / "failure_cases.csv", index=False)
                mlflow.log_artifact(str(Path(tmp) / "failure_cases.csv"))
            mlflow.log_metric("failed_checks", len(summary))
            mlflow.set_tag("valid", "false")
            print("данные не прошли проверку, отчёт failure_cases.csv в MLflow:")
            print(summary.to_string(index=False))
            return 1
        mlflow.set_tag("valid", "true")

    result = {"valid": True, "data_md5": md5, "rows": len(df)}
    print(json.dumps(result, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())