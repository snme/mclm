#!/usr/bin/env bash
# One-shot fetch of CDVAE/CrystaLLM-format benchmark splits.
#
# Source: https://github.com/lantunes/CrystaLLM (Antunes et al., Nat Commun 2024).
# Pulls the four CSP benchmarks (perov_5, carbon_24, mp_20, mpts_52) into
# /home/sathyae/orcd/pool/eval_data/csp/ in CrystaLLM's directory layout. The
# files are CSV with `cif_string` column — directly parseable by pymatgen.
#
# Idempotent: rsync --update only copies missing/newer files. Re-run after a
# CrystaLLM upstream change to refresh.

set -euo pipefail

DEST="${1:-/home/sathyae/orcd/pool/eval_data/csp}"
mkdir -p "$DEST"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "[download] cloning CrystaLLM into $TMP/crystallm (depth=1) ..."
git clone --depth=1 https://github.com/lantunes/CrystaLLM.git "$TMP/crystallm"

# CrystaLLM ships the CDVAE-format CSV splits (train/val/test) under
# resources/benchmarks/<bench>/. Each CSV has a `cif` column.
SRC_ROOT="$TMP/crystallm/resources/benchmarks"
if [[ ! -d "$SRC_ROOT" ]]; then
  echo "[download] error: $SRC_ROOT not found — repo layout may have changed." >&2
  exit 1
fi

for bench in mp_20 mpts_52 perov_5 carbon_24; do
  src="$SRC_ROOT/$bench"
  if [[ -d "$src" ]]; then
    echo "[download] $bench → $DEST/$bench/"
    mkdir -p "$DEST/$bench"
    rsync -av --update "$src/" "$DEST/$bench/"
  else
    echo "[download] (skip) $bench not found in $SRC_ROOT/"
  fi
done

echo
echo "[download] benchmarks written to $DEST"
ls -lh "$DEST"
echo
echo "[download] sanity check — first row of mp_20/test.csv (if present):"
if [[ -f "$DEST/mp_20/test.csv" ]]; then
  head -2 "$DEST/mp_20/test.csv" | cut -c -200
fi
