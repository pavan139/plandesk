#!/usr/bin/env bash
# Download the IRS publications this project indexes (public domain).
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
for f in p590a--2024 p590a--2025 p590b--2024 p590b--2025 p560--2024 p560--2025; do
  curl -sfL -o "$f.pdf" "https://www.irs.gov/pub/irs-prior/$f.pdf"
  echo "fetched $f.pdf"
done
