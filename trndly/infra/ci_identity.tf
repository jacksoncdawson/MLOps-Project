# Phase 2 — CI deploy identity (plan §8, handoff step 9).
#
# Keyless GitHub Actions → GCP auth via Workload Identity Federation (no
# long-lived JSON key, no deprecated FIREBASE_TOKEN). A dedicated deploy SA
# holds ONLY roles/firebasehosting.admin; GitHub Actions in THIS repo
# impersonate it through an OIDC-trusted WIF pool.
#
# SECURITY: the provider's attribute_condition pins the trust to exactly
# `jacksoncdawson/trndly`. Without it, ANY GitHub repository could mint tokens
# for this pool and impersonate the deploy SA — the canonical WIF footgun.
#
# This is a 💲 apply (creates the pool/provider/SA + an IAM grant). The user
# chooses TF vs a one-time console setup (handoff step 9). If applied, feed the
# two outputs below into the GitHub repo as Actions variables WIF_PROVIDER and
# DEPLOY_SA (see docs/runbooks/deploy-hosting.md).

locals {
  github_repo = "jacksoncdawson/trndly"
}

resource "google_service_account" "github_deploy" {
  project      = var.project_id
  account_id   = "sa-github-deploy"
  display_name = "GitHub Actions — Firebase Hosting deploy (WIF, keyless)"

  depends_on = [google_project_service.apis]
}

# Least privilege: Hosting deploys only. NOT roles/firebase.admin.
resource "google_project_iam_member" "github_deploy_hosting_admin" {
  project = var.project_id
  role    = "roles/firebasehosting.admin"
  member  = "serviceAccount:${google_service_account.github_deploy.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"
  description               = "OIDC federation for GitHub Actions deploys."

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  # Trust ONLY this repo's tokens. (Provider-version note: the
  # attribute_condition argument requires google provider >= 4.x; it is present
  # in the pinned ~> 6.0.)
  #
  # TODO(hardening): today ANY workflow on any branch of this repo can deploy.
  # Pin the branch too — map "attribute.ref" = "assertion.ref" above and append
  # `&& assertion.ref == 'refs/heads/main'` here — so a PR branch cannot ship
  # to Hosting. Needs a `terraform apply`; deferred while project billing is
  # detached.
  attribute_condition = "assertion.repository == '${local.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Let workflows in this repo impersonate the deploy SA. principalSet scopes the
# grant to the repo via the mapped `attribute.repository`.
resource "google_service_account_iam_member" "github_wif_user" {
  service_account_id = google_service_account.github_deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${local.github_repo}"
}

# --- Outputs to wire into GitHub repo Actions variables ---

output "ci_deploy_sa_email" {
  description = "Deploy SA email → set as GitHub Actions variable DEPLOY_SA."
  value       = google_service_account.github_deploy.email
}

output "ci_wif_provider" {
  description = "Full WIF provider resource name → set as GitHub Actions variable WIF_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.github.name
}
