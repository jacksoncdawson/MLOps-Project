from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    FEATURE_VECTOR_COLUMNS,
    TARGET_COLUMN_DEFAULT,
    TIMEFRAMES,
    prepare_training_frame,
)


def parse_args() -> argparse.Namespace:
    synthetic_dir = Path(__file__).resolve().parent / "synthetic_data"
    parser = argparse.ArgumentParser(
        description=(
            "Train and register a listing timeline classifier using local synthetic "
            "train/val/test splits."
        )
    )
    parser.add_argument(
        "--train-data-uri",
        "--data-uri",
        dest="train_data_uri",
        default=os.getenv("TRAIN_DATA_URI", str(synthetic_dir / "train.csv")),
        help="Train split CSV location (local path or gs:// URI).",
    )
    parser.add_argument(
        "--val-data-uri",
        default=os.getenv("VAL_DATA_URI", str(synthetic_dir / "val.csv")),
        help="Validation split CSV location (local path or gs:// URI).",
    )
    parser.add_argument(
        "--test-data-uri",
        default=os.getenv("TEST_DATA_URI", str(synthetic_dir / "test.csv")),
        help="Test split CSV location (local path or gs:// URI).",
    )
    parser.add_argument(
        "--target-column",
        default=os.getenv("TARGET_COLUMN", TARGET_COLUMN_DEFAULT),
        help="Target column name in the train/val/test datasets.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
        help="Random state used by the classifier.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=int(os.getenv("N_ESTIMATORS", "300")),
        help="Number of trees for RandomForestClassifier.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-team-project"),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "listing_timeline"),
        help="Registered model name in MLflow Model Registry.",
    )
    parser.add_argument(
        "--model-alias",
        default=os.getenv("MLFLOW_MODEL_ALIAS", "champion"),
        help="Alias assigned to the newly registered model.",
    )
    parser.add_argument(
        "--model-artifact-path",
        default=os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model"),
        help="Run artifact path used for the logged model.",
    )
    parser.add_argument(
        "--data-version",
        default=os.getenv("DATA_VERSION", "current-signal-v1"),
        help="Logical data version tag for reproducibility.",
    )
    return parser.parse_args()


def load_dataset(data_uri: str, split_name: str) -> pd.DataFrame:
    if not data_uri:
        raise ValueError(f"No dataset path provided for {split_name} split.")

    frame = pd.read_csv(data_uri)
    if frame.empty:
        raise ValueError(f"{split_name.capitalize()} dataset at '{data_uri}' is empty.")
    return frame


def prepare_split(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    prepared = prepare_training_frame(frame=frame, target_column=target_column)
    x = prepared[FEATURE_VECTOR_COLUMNS].copy()
    y = prepared[target_column].copy()
    return x, y


def evaluate_split(
    model: RandomForestClassifier,
    x: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict[str, float]:
    predictions = model.predict(x)
    return {
        f"{split_name}_accuracy": float(accuracy_score(y, predictions)),
        f"{split_name}_f1_weighted": float(
            f1_score(y, predictions, average="weighted", labels=TIMEFRAMES)
        ),
    }


def wait_until_model_version_ready(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    max_wait_seconds: int = 60,
) -> None:
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        current = client.get_model_version(name=model_name, version=model_version)
        if current.status == "READY":
            return
        if current.status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"Model registration failed for {model_name} version {model_version}."
            )
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for {model_name} version {model_version} to become READY."
    )


def _log_classification_report(y_true: pd.Series, y_pred: pd.Series) -> None:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=TIMEFRAMES,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_classification_report.csv",
        delete=False,
    ) as handle:
        report_path = Path(handle.name)
    report_frame.to_csv(report_path, index=True)
    mlflow.log_artifact(str(report_path), artifact_path="evaluation")
    report_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    train_frame = load_dataset(args.train_data_uri, "train")
    val_frame = load_dataset(args.val_data_uri, "validation")
    test_frame = load_dataset(args.test_data_uri, "test")

    x_train, y_train = prepare_split(train_frame, args.target_column)
    x_val, y_val = prepare_split(val_frame, args.target_column)
    x_test, y_test = prepare_split(test_frame, args.target_column)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    val_metrics = evaluate_split(model=model, x=x_val, y=y_val, split_name="val")
    test_metrics = evaluate_split(model=model, x=x_test, y=y_test, split_name="test")
    test_predictions = pd.Series(model.predict(x_test), index=y_test.index)

    with mlflow.start_run(run_name="listing_timeline_timeframe_classifier") as run:
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "target_column": args.target_column,
                "feature_count": len(FEATURE_VECTOR_COLUMNS),
                "random_state": args.random_state,
                "n_estimators": args.n_estimators,
                "train_data_uri": args.train_data_uri,
                "val_data_uri": args.val_data_uri,
                "test_data_uri": args.test_data_uri,
                "data_version": args.data_version,
                "timeframes": ",".join(TIMEFRAMES),
            }
        )
        mlflow.log_metrics({**val_metrics, **test_metrics})
        mlflow.set_tag("registered_model_name", args.registered_model_name)
        mlflow.set_tag("model_alias", args.model_alias)
        mlflow.set_tag("problem_type", "multiclass_classification")

        _log_classification_report(y_true=y_test, y_pred=test_predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=args.model_artifact_path,
            input_example=x_train.head(1),
        )

        run_id = run.info.run_id

    source_model_uri = f"runs:/{run_id}/{args.model_artifact_path}"
    model_version = mlflow.register_model(
        model_uri=source_model_uri,
        name=args.registered_model_name,
    )

    client = MlflowClient(tracking_uri=args.tracking_uri)
    wait_until_model_version_ready(
        client=client,
        model_name=args.registered_model_name,
        model_version=model_version.version,
    )
    client.set_registered_model_alias(
        name=args.registered_model_name,
        alias=args.model_alias,
        version=model_version.version,
    )

    print("Training run complete.")
    print(f"Run ID: {run_id}")
    print(f"Validation accuracy: {val_metrics['val_accuracy']:.4f}")
    print(f"Validation weighted F1: {val_metrics['val_f1_weighted']:.4f}")
    print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test weighted F1: {test_metrics['test_f1_weighted']:.4f}")
    print(
        "Registered model URI: "
        f"models:/{args.registered_model_name}@{args.model_alias} "
        f"(version {model_version.version})"
    )


if __name__ == "__main__":
    main()
