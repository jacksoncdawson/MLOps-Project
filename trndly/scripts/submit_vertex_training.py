from google.cloud import aiplatform

aiplatform.init(project="ml-ops-491417", location="us-central1", staging_bucket="gs://trndly-mlops-us")

job = aiplatform.CustomContainerTrainingJob(
    display_name="trndly-listing-timeline-v1",
    container_uri="us-central1-docker.pkg.dev/ml-ops-491417/trndly-repo/trndly-trainer:v1",
    model_serving_container_image_uri="us-central1-docker.pkg.dev/ml-ops-491417/trndly-repo/trndly-api:v1",
)

model = job.run(
    machine_type="n1-standard-4",   # overkill for RF, but fast; n1-standard-2 works too
    replica_count=1,
    environment_variables={
        "MLFLOW_TRACKING_URI": "gs://trndly-mlops-us/mlflow",
        "TRAIN_DATA_URI": "gs://trndly-mlops-us/data/synthetic/train.csv",
        "VAL_DATA_URI": "gs://trndly-mlops-us/data/synthetic/val.csv",
        "TEST_DATA_URI": "gs://trndly-mlops-us/data/synthetic/test.csv",
        "TREND_SIGNALS_PATH": "gs://trndly-mlops-us/data/synthetic/trend_signals.csv",
    },
    service_account="trndly-vertex@ml-ops-491417.iam.gserviceaccount.com",
    base_output_dir="gs://trndly-mlops-us/training-jobs/v1",
    sync=True,  # block until done (RF on synthetic data: < 3 min)
)