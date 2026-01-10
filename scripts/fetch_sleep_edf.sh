#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-"$(pwd)/data/raw/sleep-edfx-1.0.0"}
REMOTE_ROOT=${SLEEP_EDF_REMOTE_ROOT:-"sleep-edfx/1.0.0"}
SUBSET=${SLEEP_EDF_SUBSET:-"sleep-telemetry"}
PHYSIONET_BASE_URL=${PHYSIONET_BASE_URL:-"https://physionet.org/files/${REMOTE_ROOT}"}

log() { echo "[sleep-stager] $*"; }
die() { echo "[sleep-stager] $*" >&2; exit 1; }

mkdir -p "${DATA_DIR}"

if ! command -v wfdbdownload >/dev/null 2>&1; then
  die "wfdbdownload not found; install WFDB (e.g. pip install wfdb or system package) and retry."
fi

log "Downloading ${REMOTE_ROOT}/${SUBSET} into ${DATA_DIR}"
wfdbdownload -r "${REMOTE_ROOT}/${SUBSET}" -p "${DATA_DIR}"

nested_subset="${DATA_DIR}/${REMOTE_ROOT}/${SUBSET}"
if [ ! -d "${DATA_DIR}/${SUBSET}" ] && [ -d "${nested_subset}" ]; then
  log "Relocating ${nested_subset} to ${DATA_DIR}/${SUBSET}"
  mkdir -p "${DATA_DIR}"
  mv "${nested_subset}" "${DATA_DIR}/${SUBSET}"
  nested_root="${DATA_DIR}/${REMOTE_ROOT}"
  if [ -f "${nested_root}/SHA256SUMS.txt" ] && [ ! -f "${DATA_DIR}/SHA256SUMS.txt" ]; then
    mv "${nested_root}/SHA256SUMS.txt" "${DATA_DIR}/SHA256SUMS.txt"
  fi
  rmdir -p "${nested_root}" 2>/dev/null || true
fi

sha_file="${DATA_DIR}/SHA256SUMS.txt"
if [ ! -f "${sha_file}" ]; then
  log "Downloading SHA256SUMS.txt"
  if command -v curl >/dev/null 2>&1; then
    auth_args=()
    if [ -n "${PHYSIONET_USERNAME:-}" ] && [ -n "${PHYSIONET_PASSWORD:-}" ]; then
      auth_args=(-u "${PHYSIONET_USERNAME}:${PHYSIONET_PASSWORD}")
    fi
    curl -fL "${auth_args[@]}" "${PHYSIONET_BASE_URL}/SHA256SUMS.txt" -o "${sha_file}"
  elif command -v wget >/dev/null 2>&1; then
    auth_args=()
    if [ -n "${PHYSIONET_USERNAME:-}" ] && [ -n "${PHYSIONET_PASSWORD:-}" ]; then
      auth_args=(--user "${PHYSIONET_USERNAME}" --password "${PHYSIONET_PASSWORD}")
    fi
    wget "${auth_args[@]}" -O "${sha_file}" "${PHYSIONET_BASE_URL}/SHA256SUMS.txt"
  else
    die "curl or wget required to download SHA256SUMS.txt."
  fi
fi

log "Verifying checksums for ${SUBSET}"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "${DATA_DIR}" && grep "${SUBSET}/" SHA256SUMS.txt | sha256sum -c -)
elif command -v shasum >/dev/null 2>&1; then
  (cd "${DATA_DIR}" && grep "${SUBSET}/" SHA256SUMS.txt | shasum -a 256 -c -)
else
  die "sha256sum or shasum not found; cannot verify checksums."
fi

log "Download and verification complete."
