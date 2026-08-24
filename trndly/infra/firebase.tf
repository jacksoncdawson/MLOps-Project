# Phase 2 — Static serving (plan §6/§8). Firebase project + Hosting site.
#
# These three resources are the ONLY ones that require the beta provider
# (plan §6 "provider split"); everything else in this module stays on stable
# `google`. The SPA + canonical JSON are deployed to this site via
# `firebase deploy` (CLI/CI) — Terraform provisions the site, never content.
#
# IRREVERSIBLE: enabling Firebase on the project (google_firebase_project)
# cannot be undone. site_id is immutable once created. See the apply gate in
# docs/handoff-phase2-3.md / docs/runbooks/deploy-hosting.md.

resource "google_firebase_project" "default" {
  provider = google-beta
  project  = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_firebase_web_app" "default" {
  provider     = google-beta
  project      = var.project_id
  display_name = "trndly"

  # DELETE removes the web app on `terraform destroy`. The alternative,
  # ABANDON, keeps it — but Firebase only soft-deletes apps, and a lingering
  # soft-deleted app can conflict with recreating the same app later, so a
  # destroyed stack should not leave one behind. (An earlier comment here
  # described ABANDON while the value said DELETE; the value is the intent.)
  deletion_policy = "DELETE"

  depends_on = [google_firebase_project.default]
}

# site_id "trndly" is a deliberate brand label (≠ project id) → URL
# https://trndly.web.app. Immutable. Confirm it is free at apply time
# (`firebase hosting:sites:create trndly`); fall back to trndly-app / gettrndly
# if taken. If a default site already exists at this id, import it before apply:
#   terraform import google_firebase_hosting_site.default <project>/trndly
resource "google_firebase_hosting_site" "default" {
  provider = google-beta
  project  = var.project_id
  site_id  = "trndly"
  app_id   = google_firebase_web_app.default.app_id

  depends_on = [google_firebase_project.default]
}
