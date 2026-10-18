from datetime import UTC, datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import CronTriggerTimetable, Param, dag, task
from kubernetes.client import models as k8s

PULL = "uv run --no-sync dvc pull -r cluster datasets\bank_additional_pre_quickstart.csv && "
COMMON = dict(namespace = "mlops", in_cluster = True, 
        image = "{{ var.value.bank_image }}", image_pull_policy = "IfNotPresent",
        get_logs = True, on_finish_action = "delete_pod")

@dag(
    schedule = CronTriggerTimetable("0 3 * * *", timezone = "Europe/Moscow"),
    catchup=False,
    params = {
        "DEPTH": Param(6, type="integer", description="Глубина деревьев в Catboost"),
    },
    tags = ["bank"]
)
def bank_retrain():
    validate = KubernetesPodOperator(
        task_id = "validate",
        name = "bank-validate",
        env_vars = {
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops:5000",
            "DATA_PATH": "datasets/bank_additional_pre_quickstart.csv"
        },
        env_from = [
            k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="s3-credentials"))
        ],
        cmds = ["sh", "-c"],
        arguments = [
        (
            PULL + "uv run --no-sync python -m bank.data"
        )
        ],
        **COMMON
    )

    train = KubernetesPodOperator(
        task_id = "train",
        name = "bank-train",
        cmds = ["sh", "-c"],
        arguments = [
        (
            PULL + "uv run --no-sync python -m bank.train"
        )
        ],
        env_vars = {
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops:5000",
            "DATA_PATH": "datasets/bank_additional_pre_quickstart.csv",
            "DEPTH": "{{ params.DEPTH }}"
        },
        env_from = [
            k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="s3-credentials"))
        ],
        do_xcom_push = True,
        **COMMON
    )


    @task.short_circuit
    def promoted(result: dict) -> bool:
        print(f"версия {result['version']}")
        print(f"pr_auc {result['pr_auc']}") 
        print(f"pr_auc до этого у champion {result['champion_pr_auc_before']}")
        return result['promoted']

    @task
    def rollout():
        from kubernetes import client, config

        config.load_incluster_config()
        now = datetime.now(UTC).isoformat()
        client.AppsV1Api().patch_namespaced_deployment(
            "bank-service", "default",
            {"spec": {"template": {"metadata": {"annotations": {
                "kubectl.kubernetes.io/restartedAt": now}}}}}
        )
        print("Инициирован перезапуск bank-service", now)
    
    validate >> train
    promoted(train.output) >> rollout()


bank_retrain()
