#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stage a model directory from (slow, shared) storage to node-local storage.
#
# Runs once per allocated worker node (srun --ntasks-per-node=1) inside the
# worker container, BEFORE workers start. A per-destination lock serializes
# concurrent jobs. Files are copied into a private temporary directory and are
# made visible only after verification and a completion marker are written.
# Re-runs skip a cache whose source identity and destination manifest match.
# Symlinks are dereferenced, so the staged tree is self-contained. On any
# failure the script exits non-zero and workers never fall back to shared
# storage.
#
# Usage: stage_model.sh <SOURCE_DIR> <DEST_DIR>
#   SOURCE_DIR  in-container path of the shared model (srtctl mounts it at /model)
#   DEST_DIR    node-local path to stage into (e.g. /raid/scratch/models/<name>)
#
# Env:
#   STAGE_PARALLEL  per-node copy fan-out (default 16)
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: stage_model.sh <SOURCE_DIR> <DEST_DIR>" >&2
    exit 2
fi
SRC="$1"
DEST="$2"
PARALLEL="${STAGE_PARALLEL:-16}"
HOST="$(hostname)"
MARKER=".srtctl-stage-complete"

if [ ! -d "$SRC" ]; then
    echo "[stage:$HOST] ERROR: source dir not found: $SRC" >&2
    exit 1
fi

case "$DEST" in
    "/")
        echo "[stage:$HOST] ERROR: destination must be an absolute, non-root path: $DEST" >&2
        exit 2
        ;;
    /*) ;;
    *)
        echo "[stage:$HOST] ERROR: destination must be an absolute, non-root path: $DEST" >&2
        exit 2
        ;;
esac
SRC_REAL="$(readlink -f -- "$SRC")"
DEST_REAL="$(readlink -m -- "$DEST")"
if [[ "$DEST_REAL" == "$SRC_REAL" || "$DEST_REAL" == "$SRC_REAL/"* || "$SRC_REAL" == "$DEST_REAL/"* ]]; then
    echo "[stage:$HOST] ERROR: source and destination must not overlap: $SRC -> $DEST" >&2
    exit 2
fi
if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "[stage:$HOST] ERROR: STAGE_PARALLEL must be a positive integer: $PARALLEL" >&2
    exit 2
fi

DEST_PARENT="$(dirname -- "$DEST")"
DEST_BASE="$(basename -- "$DEST")"
mkdir -p "$DEST_PARENT"

LOCK="${DEST_PARENT}/.${DEST_BASE}.stage.lock"
exec 9>"$LOCK"
echo "[stage:$HOST] waiting for cache lock $LOCK"
flock -x 9

WORK_DIR="$(mktemp -d "${DEST_PARENT}/.${DEST_BASE}.stage-work.XXXXXX")"
SOURCE_META="${WORK_DIR}/source.meta"
SOURCE_CONTENT="${WORK_DIR}/source.content"
DEST_CONTENT="${WORK_DIR}/destination.content"
TMP=""
cleanup() {
    rm -rf -- "$WORK_DIR"
    if [ -n "$TMP" ] && [ -d "$TMP" ]; then
        rm -rf -- "$TMP"
    fi
}
trap cleanup EXIT

# The source identity is cheap to compute even for multi-terabyte checkpoints:
# path, size, and nanosecond mtime for every dereferenced regular file. It is
# stronger than the old name+size hit check without rereading all weight bytes.
( cd "$SRC" && find -L . -type f -printf '%P\t%s\t%T@\n' 2>/dev/null | LC_ALL=C sort ) > "$SOURCE_META"
SOURCE_ID="$(sha256sum "$SOURCE_META" | awk '{print $1}')"
( cd "$SRC" && find -L . -type f -printf '%P\t%s\n' 2>/dev/null | LC_ALL=C sort ) > "$SOURCE_CONTENT"
SOURCE_BYTES="$(awk -F '\t' '{sum += $2} END {printf "%.0f\n", sum}' "$SOURCE_CONTENT")"

cache_matches() {
    [ -f "$DEST/$MARKER" ] || return 1
    [ "$(cat "$DEST/$MARKER")" = "$SOURCE_ID" ] || return 1
    ( cd "$DEST" && find -L . -type f ! -name "$MARKER" -printf '%P\t%s\n' 2>/dev/null | LC_ALL=C sort ) \
        > "$DEST_CONTENT"
    diff -q "$SOURCE_CONTENT" "$DEST_CONTENT" >/dev/null 2>&1
}

if cache_matches; then
    echo "[stage:$HOST] $DEST manifest hit (source_id=$SOURCE_ID) — skipping copy"
    exit 0
fi

AVAILABLE_BYTES="$(df -Pk "$DEST_PARENT" | awk 'NR == 2 {printf "%.0f\n", $4 * 1024}')"
if [ -z "$AVAILABLE_BYTES" ] || [ "$AVAILABLE_BYTES" -lt "$SOURCE_BYTES" ]; then
    echo "[stage:$HOST] ERROR: insufficient free space under $DEST_PARENT" >&2
    echo "[stage:$HOST] required=$SOURCE_BYTES available=${AVAILABLE_BYTES:-unknown}" >&2
    exit 1
fi

TMP="${DEST_PARENT}/.${DEST_BASE}.stage-${SOURCE_ID}"
if [ -e "$TMP" ]; then
    rm -rf -- "$TMP"
fi
mkdir -p "$TMP"

echo "[stage:$HOST] staging $SRC -> $DEST (parallel=$PARALLEL source_id=$SOURCE_ID bytes=$SOURCE_BYTES)"
start=$(date +%s)
( cd "$SRC" && find -L . -type f -print0 \
    | xargs -0 -P "$PARALLEL" -I{} bash -c '
        rel="$1"; dest_root="$2"
        mkdir -p "$dest_root/$(dirname "$rel")"
        cp -L -- "$rel" "$dest_root/$rel"
      ' _ {} "$TMP" )

# Verify the private tree before publishing it.
( cd "$TMP" && find -L . -type f -printf '%P\t%s\n' 2>/dev/null | LC_ALL=C sort ) > "$DEST_CONTENT"
if ! diff -q "$SOURCE_CONTENT" "$DEST_CONTENT" >/dev/null 2>&1; then
    echo "[stage:$HOST] ERROR: manifest mismatch after copy" >&2
    exit 1
fi
printf '%s\n' "$SOURCE_ID" > "$TMP/$MARKER"

# The destination is a disposable node-local cache. Replace a stale generation
# only after the new generation is complete; the enclosing Slurm allocation is
# exclusive, and the lock prevents another staging process from racing us.
if [ -e "$DEST" ]; then
    rm -rf -- "$DEST"
fi
mv -- "$TMP" "$DEST"
TMP=""

echo "[stage:$HOST] done in $(( $(date +%s) - start ))s; manifest verified (source_id=$SOURCE_ID)"
