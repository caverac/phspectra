#!/usr/bin/env bash
# Publish a new version of the phspectra source distribution on Zenodo.
#
# Required env:
#   ZENODO_TOKEN       Personal access token with deposit:write + deposit:actions scopes.
#   ZENODO_CONCEPT_ID  Integer concept-record id (the number after "zenodo." in the
#                      concept DOI: 10.5281/zenodo.<ID>). If unset, the script
#                      exits 0 without doing anything.
#   VERSION            Semantic version of the release being archived (e.g. 1.0.3).
#   SDIST_PATH         Path to the sdist .tar.gz to upload.
#
# Exit codes:
#   0  success, or ZENODO_CONCEPT_ID unset (no-op).
#   1  required env missing or upload failed.

set -euo pipefail

: "${ZENODO_TOKEN:?ZENODO_TOKEN is required}"
: "${VERSION:?VERSION is required}"
: "${SDIST_PATH:?SDIST_PATH is required}"

if [[ -z "${ZENODO_CONCEPT_ID:-}" ]]; then
  echo "ZENODO_CONCEPT_ID is not set; skipping Zenodo archival." >&2
  exit 0
fi

if [[ ! -f "$SDIST_PATH" ]]; then
  echo "sdist not found at $SDIST_PATH" >&2
  exit 1
fi

API="https://zenodo.org/api"
AUTH=(-H "Authorization: Bearer ${ZENODO_TOKEN}")
TODAY="$(date -u +%Y-%m-%d)"
SDIST_NAME="$(basename "$SDIST_PATH")"

echo "Looking up latest published version of concept ${ZENODO_CONCEPT_ID}..."
LATEST_ID="$(curl -fsSL "${API}/records/${ZENODO_CONCEPT_ID}/versions/latest" \
              | jq -r '.id')"
echo "  latest version id: ${LATEST_ID}"

echo "Creating new version draft..."
NEWVER="$(curl -fsSL -X POST "${AUTH[@]}" \
            "${API}/deposit/depositions/${LATEST_ID}/actions/newversion")"
DRAFT_URL="$(echo "${NEWVER}" | jq -r '.links.latest_draft')"
DRAFT_ID="${DRAFT_URL##*/}"
echo "  draft id: ${DRAFT_ID}"

DRAFT="$(curl -fsSL "${AUTH[@]}" "${API}/deposit/depositions/${DRAFT_ID}")"
BUCKET_URL="$(echo "${DRAFT}" | jq -r '.links.bucket')"

echo "Removing files inherited from previous version..."
echo "${DRAFT}" | jq -r '.files[].id' | while read -r FILE_ID; do
  [[ -z "${FILE_ID}" ]] && continue
  curl -fsSL -X DELETE "${AUTH[@]}" \
       "${API}/deposit/depositions/${DRAFT_ID}/files/${FILE_ID}" >/dev/null
done

echo "Uploading ${SDIST_NAME}..."
curl -fsSL --upload-file "${SDIST_PATH}" "${AUTH[@]}" \
     "${BUCKET_URL}/${SDIST_NAME}" >/dev/null

echo "Updating version metadata to ${VERSION} (publication date ${TODAY})..."
META="$(echo "${DRAFT}" | jq --arg version "${VERSION}" --arg pubdate "${TODAY}" \
          '{metadata: (.metadata | .version=$version | .publication_date=$pubdate)}')"
curl -fsSL -X PUT "${AUTH[@]}" \
     -H "Content-Type: application/json" \
     -d "${META}" \
     "${API}/deposit/depositions/${DRAFT_ID}" >/dev/null

echo "Publishing..."
PUBLISHED="$(curl -fsSL -X POST "${AUTH[@]}" \
              "${API}/deposit/depositions/${DRAFT_ID}/actions/publish")"
NEW_DOI="$(echo "${PUBLISHED}" | jq -r '.doi')"
echo "Published: https://doi.org/${NEW_DOI}"
