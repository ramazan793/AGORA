#!/usr/bin/env bash
#
# setup_cuda_deps.sh — install AGORA's CUDA-compiled extensions into the uv venv.
#
# These extensions are NOT resolved by uv (they must be compiled against your CUDA
# toolkit; we used CUDA 11.8). This helper implements the "fast path": copy the
# already-compiled packages from an existing environment that has them built for the
# SAME Python / PyTorch / CUDA ABI (e.g. a GGHead/EG3D conda env or another venv).
#
# To build from source instead, see the table in README.md ("CUDA extensions").
#
# Usage:
#   SRC=/path/to/existing/site-packages bash scripts/setup_cuda_deps.sh
#
# SRC  — site-packages of an environment that already has the forks compiled.
# DST  — defaults to this repo's .venv site-packages; override by exporting DST.
#
set -euo pipefail

# Resolve the repo root from this script's location (scripts/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DST="${DST:-$REPO_ROOT/.venv/lib/python3.10/site-packages}"

if [ -z "${SRC:-}" ]; then
  echo "ERROR: set SRC to an existing site-packages that has the compiled forks." >&2
  echo "  e.g. SRC=/path/to/env/lib/python3.10/site-packages bash scripts/setup_cuda_deps.sh" >&2
  exit 1
fi

if [ ! -d "$SRC" ]; then echo "ERROR: SRC does not exist: $SRC" >&2; exit 1; fi
if [ ! -d "$DST" ]; then echo "ERROR: DST does not exist (create the venv first): $DST" >&2; exit 1; fi

echo "SRC = $SRC"
echo "DST = $DST"

PKGS=(
  gsplat
  pytorch3d
  simple_knn
  diff_gaussian_rasterization
  diff_gaussian_rasterization_features
  diff_gaussian_rasterization_radegs
  diff_gaussian_rasterization_distwar
  diff_gaussian_rasterization_distwar_features
  gaussian_splatting
  eg3d
  nvdiffrast
)

for p in "${PKGS[@]}"; do
  if [ -d "$SRC/$p" ]; then
    cp -rp "$SRC/$p" "$DST/"
    echo "copied dir: $p"
  else
    echo "MISSING dir (skipped): $p"
  fi
  # copy matching dist-info / egg-info metadata so pip/uv see the package as installed
  for meta in "$SRC/$p"-*.dist-info "$SRC/$p"-*.egg-info; do
    if [ -e "$meta" ]; then
      cp -rp "$meta" "$DST/"
      echo "  copied meta: $(basename "$meta")"
    fi
  done
done

echo "=== done ==="
echo "Verify with: uv run --no-sync python -c 'import gsplat, pytorch3d, eg3d, simple_knn, gaussian_splatting; print(\"forks OK\")'"
echo "NOTE: a bare 'uv sync' will prune these again — use 'uv run --no-sync' / 'uv sync --inexact'."
