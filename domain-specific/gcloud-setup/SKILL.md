---
name: gcloud-setup
description: Install and authenticate the Google Cloud CLI (gcloud, gsutil, bq) on macOS. Use when the user needs to set up gcloud from scratch, re-authenticate, add OAuth scopes for a specific Google API (Search Console, Drive, Sheets, etc.), or fix Application Default Credentials (ADC) issues.
author: Greg Z.
author_email: info@extractum.io
author_url: https://www.linkedin.com/in/gregzem/
---

# gcloud Setup & Authentication

This skill is the canonical setup path for `gcloud` on macOS. It uses **only commands verified to work** — no guessing, no deprecated flags.

## When to use this skill

- `gcloud`, `gsutil`, or `bq` is not installed.
- A Google API call returns `PERMISSION_DENIED` due to missing scopes or quota project.
- ADC (Application Default Credentials) needs to be (re)created with custom scopes for a non-default Google API.
- User says something like "set up gcloud", "authenticate to Google Cloud", "fix ADC", "add webmasters/drive/sheets scope".

## Key facts (memorize — these were the gotchas)

1. **The Homebrew cask is `gcloud-cli`**, NOT `google-cloud-sdk` (the old name is deprecated).
2. **`gcloud auth login` does NOT accept `--scopes`.** For custom scopes use `gcloud auth application-default login --scopes=...`.
3. **`application-default login` REQUIRES `cloud-platform` in the scope list**, even if the target API doesn't need it. Omitting it produces: `https://www.googleapis.com/auth/cloud-platform scope is required but not requested`.
4. **All scope boxes must be checked on the Google consent screen.** If the user unchecks `cloud-platform`, the login succeeds but ADC is unusable: `scope is required but not consented`.
5. **User OAuth credentials need a quota project** on API calls via the `x-goog-user-project` header. Without it: `accessNotConfigured` / `SERVICE_DISABLED`.
6. **Two different token commands exist.** `gcloud auth print-access-token` returns the token for the user-login (fixed gcloud scopes). `gcloud auth application-default print-access-token` returns the ADC token (custom scopes). **For calling third-party Google APIs with custom scopes, always use the ADC token.**

## Step 1: Install

```bash
# Check first
which gcloud gsutil bq

# If missing, install via Homebrew cask
brew install --cask gcloud-cli

# Verify
gcloud --version
bq version
gsutil version
```

Python note: the cask bundles a venv; if it warns about Python 3.9, fix with:
```bash
brew install python@3.12
export CLOUDSDK_PYTHON="$(brew --prefix)/opt/python@3.12/bin/python3.12"
gcloud components reinstall
```

## Step 2: User login (for running `gcloud` commands)

```bash
gcloud auth login
```

This opens a browser. The resulting credentials are used by `gcloud` CLI commands (project management, resource listing, etc.) but **do NOT carry custom API scopes** — they only have the fixed gcloud default scopes.

## Step 3: Pick a project and enable the API you need

```bash
# See projects
gcloud projects list

# Set the default
gcloud config set project YOUR_PROJECT_ID

# Enable the API(s) you intend to call. Examples:
gcloud services enable searchconsole.googleapis.com
gcloud services enable drive.googleapis.com
gcloud services enable sheets.googleapis.com
gcloud services enable bigquery.googleapis.com
```

## Step 4: ADC with custom scopes (for calling Google APIs via curl or SDK)

This is the step most people get wrong. Use **application-default login**, not plain login, and include `cloud-platform` plus the target-API scope.

Build the scope string with explicit components to avoid quoting errors:

```bash
SCOPES="openid"
SCOPES="$SCOPES,https://www.googleapis.com/auth/userinfo.email"
SCOPES="$SCOPES,https://www.googleapis.com/auth/cloud-platform"   # REQUIRED
SCOPES="$SCOPES,https://www.googleapis.com/auth/webmasters.readonly"   # or your target scope
gcloud auth application-default login --scopes="$SCOPES"
```

**When the browser opens: check every scope box on the consent page.** If `cloud-platform` is unchecked, the login will succeed but any subsequent API call fails.

### Common target scopes (add to `SCOPES` as needed)

| API | Scope |
|---|---|
| Search Console (read) | `https://www.googleapis.com/auth/webmasters.readonly` |
| Search Console (read/write) | `https://www.googleapis.com/auth/webmasters` |
| Drive (read) | `https://www.googleapis.com/auth/drive.readonly` |
| Sheets | `https://www.googleapis.com/auth/spreadsheets` |
| Gmail (read) | `https://www.googleapis.com/auth/gmail.readonly` |
| Calendar | `https://www.googleapis.com/auth/calendar` |
| Analytics (read) | `https://www.googleapis.com/auth/analytics.readonly` |
| YouTube (read) | `https://www.googleapis.com/auth/youtube.readonly` |

## Step 5: Call the API

Always include `x-goog-user-project` header when using user/ADC credentials — it tells Google which project gets the API quota.

```bash
TOKEN=$(gcloud auth application-default print-access-token)
PROJECT=$(gcloud config get-value project)

curl -s \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-goog-user-project: $PROJECT" \
  "https://<API>.googleapis.com/<path>" | jq
```

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `unrecognized arguments: --scopes=...` on `gcloud auth login` | Wrong command | Use `gcloud auth application-default login --scopes=...` |
| `cloud-platform scope is required but not requested` | Missing from `--scopes` | Add it to the scope list |
| `cloud-platform scope is required but not consented` | User unchecked it on consent screen | Rerun the ADC login and check **all** boxes |
| `API requires a quota project` | Missing header | Add `-H "x-goog-user-project: $PROJECT"` |
| `SERVICE_DISABLED` | API not enabled on project | `gcloud services enable <API>.googleapis.com` |
| `caller does not have permission` (on API resource) | The logged-in Google account lacks access to the resource itself | Grant access in the resource's own admin UI (e.g. Search Console → Users & permissions) |

## Service accounts (unattended use)

For cron jobs and non-interactive environments, prefer a service account:

```bash
gcloud iam service-accounts create my-sa --display-name="My SA"
gcloud iam service-accounts keys create sa-key.json \
  --iam-account=my-sa@YOUR_PROJECT.iam.gserviceaccount.com
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/sa-key.json"
```

Then **grant the service account access to the target resource in that product's own UI** (Search Console, Drive folder, Sheet, etc.) — IAM alone is not enough for most user-data APIs.

## Done criteria

The setup is complete when:
1. `gcloud --version` prints a version.
2. `gcloud config get-value project` returns a project ID (not `(unset)`).
3. `gcloud auth application-default print-access-token` prints a token starting with `ya29.`.
4. A sample API call with the quota-project header returns data (not `PERMISSION_DENIED`).
