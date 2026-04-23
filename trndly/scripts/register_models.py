import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import aiplatform

# ── config ────────────────────────────────────────────────────────────────────
PROJECT = "ml-ops-491417"
REGION = "us-central1"
TRACKING_URI = "gs://trndly-mlops-us/mlflow"
REGISTERED_NAME = "listing_timeline"
ALIAS = "champion"
SERVING_IMAGE = f"us-central1-docker.pkg.dev/{PROJECT}/trndly-repo/trndly-api:v1"
TREND_SIGNALS= "gs://trndly-mlops-us/data/synthetic/trend_signals.csv"
# ──────────────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# Look up the champion version by alias
version = client.get_model_version_by_alias(name=REGISTERED_NAME, alias=ALIAS)
run_id = version.run_id
artifact_uri = client.get_run(run_id).info.artifact_uri # gs://.../<exp>/<run>/artifacts

model_gcs_path = f"{artifact_uri}/model"
print(f"Model artifact at: {model_gcs_path}")

# Register in Vertex AI Model Registry
aiplatform.init(project=PROJECT, location=REGION, staging_bucket="gs://trndly-mlops-us")

model = aiplatform.Model.upload(
display_name="listing-timeline-classifier",
artifact_uri=model_gcs_path,
serving_container_image_uri=SERVING_IMAGE,
serving_container_predict_route="/predict",
serving_container_health_route="/health",
serving_container_ports=[{"containerPort": 8000}],
serving_container_environment_variables={
"MLFLOW_TRACKING_URI": TRACKING_URI,
"MLFLOW_MODEL_URI": f"models:/{REGISTERED_NAME}@{ALIAS}",
"TREND_SIGNALS_PATH":TREND_SIGNALS,},
labels={"alias": "champion", "version": "v1"},
)

print(f"Vertex AI model registered: {model.resource_name}")
print(f"Model display name: {model.display_name}")