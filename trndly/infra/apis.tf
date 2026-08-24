# Project APIs (plan §6, Phase 0). Enabled up front for the whole build so later
# phases can `apply` without a separate enablement step. `disable_on_destroy =
# false`: destroying one phase's resources must never tear down an API another
# phase (or another project workload) still needs.
#
# Scope note: APIs once pre-enabled for unbuilt work were REMOVED from this set
# — servicenetworking + vpcaccess (a private-IP Cloud SQL path, evaluated and
# rejected for this project: single operator, IAM-gated Auth Proxy is the
# fit-for-purpose control) and firestore + identitytoolkit (Phase 5, deferred).
# Enable them in the phase that builds them. Removing an entry here only
# destroys the TF resource; with disable_on_destroy = false the API itself
# stays enabled in the project until disabled out-of-band.
locals {
  gcp_apis = toset([
    # Enablement plumbing (required for google_project_service itself).
    "cloudresourcemanager.googleapis.com",
    "serviceusage.googleapis.com",
    "iam.googleapis.com",
    # Storage (state bucket already exists; artifacts/Hosting buckets later).
    "storage.googleapis.com",
    # Phase 3 — MLflow runtime + backend.
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    # Phase 2 — Firebase Hosting.
    "firebase.googleapis.com",
    "firebasehosting.googleapis.com",
    # Phase 2 — CI deploy via Workload Identity Federation (token exchange +
    # short-lived SA credentials for the GitHub Actions deploy job).
    "sts.googleapis.com",
    "iamcredentials.googleapis.com",
  ])
}

resource "google_project_service" "apis" {
  for_each = local.gcp_apis

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}
