#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Seed a small, fingerprinted TVM-FFI JIT cache from shared storage into the
# allocation-local NVMe before an SGLang worker starts.  srtctl runs setup_script
# once for each worker launcher; the lock also makes this safe for recipes with
# multiple launchers on one node.

set -euo pipefail

SHARED_CACHE="${TVM_FFI_SHARED_CACHE_DIR:?TVM_FFI_SHARED_CACHE_DIR is required}"
LOCAL_CACHE="${TVM_FFI_CACHE_DIR:?TVM_FFI_CACHE_DIR is required}"
MANIFEST=".srtctl-tvm-ffi-sha256"

case "${SHARED_CACHE}" in
    /*) ;;
    *) echo "ERROR: TVM_FFI_SHARED_CACHE_DIR must be absolute: ${SHARED_CACHE}" >&2; exit 2 ;;
esac
case "${LOCAL_CACHE}" in
    /|"") echo "ERROR: TVM_FFI_CACHE_DIR must be an absolute, non-root path" >&2; exit 2 ;;
    /*) ;;
    *) echo "ERROR: TVM_FFI_CACHE_DIR must be absolute: ${LOCAL_CACHE}" >&2; exit 2 ;;
esac

SHARED_REAL="$(readlink -f -- "${SHARED_CACHE}")"
LOCAL_REAL="$(readlink -m -- "${LOCAL_CACHE}")"
if [[ "${SHARED_REAL}" == "${LOCAL_REAL}" || "${SHARED_REAL}" == "${LOCAL_REAL}/"* || "${LOCAL_REAL}" == "${SHARED_REAL}/"* ]]; then
    echo "ERROR: shared and local TVM-FFI cache paths must not overlap" >&2
    exit 2
fi
if [ ! -f "${SHARED_REAL}/${MANIFEST}" ]; then
    echo "ERROR: shared TVM-FFI cache manifest is missing: ${SHARED_REAL}/${MANIFEST}" >&2
    exit 3
fi

verify_cache() {
    local cache_root="$1"
    ( cd "${cache_root}" && sha256sum -c --quiet "${MANIFEST}" )
}

# Refuse to stage a corrupt/incomplete shared generation.
verify_cache "${SHARED_REAL}"

LOCAL_PARENT="$(dirname -- "${LOCAL_REAL}")"
LOCAL_BASE="$(basename -- "${LOCAL_REAL}")"
mkdir -p "${LOCAL_PARENT}"
LOCK="${LOCAL_PARENT}/.${LOCAL_BASE}.stage.lock"
exec 9>"${LOCK}"
flock -x 9

if [ -d "${LOCAL_REAL}" ]; then
    if verify_cache "${LOCAL_REAL}"; then
        echo "TVM-FFI cache already staged on $(hostname): ${LOCAL_REAL}"
        exit 0
    fi
    echo "ERROR: refusing to replace invalid existing local TVM-FFI cache: ${LOCAL_REAL}" >&2
    exit 4
fi

STAGE_DIR="$(mktemp -d "${LOCAL_PARENT}/.${LOCAL_BASE}.stage.XXXXXX")"
cleanup() {
    if [ -n "${STAGE_DIR:-}" ] && [ -d "${STAGE_DIR}" ]; then
        rm -rf -- "${STAGE_DIR}"
    fi
}
trap cleanup EXIT

cp -a "${SHARED_REAL}/." "${STAGE_DIR}/"
verify_cache "${STAGE_DIR}"
mv -- "${STAGE_DIR}" "${LOCAL_REAL}"
STAGE_DIR=""

SO_COUNT="$(find "${LOCAL_REAL}" -type f -name '*.so' | wc -l)"
FILE_COUNT="$(find "${LOCAL_REAL}" -type f ! -name "${MANIFEST}" | wc -l)"
echo "Staged TVM-FFI cache on $(hostname): so=${SO_COUNT} files=${FILE_COUNT} local=${LOCAL_REAL}"
