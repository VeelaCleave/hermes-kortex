#!/bin/bash
# Sync hermes-kortex from repo to ~/.hermes/plugins/kortex
# Usage: bash scripts/sync-to-hermes.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="$HOME/.hermes/plugins/kortex"

echo "📦 Syncing hermes-kortex to Hermes plugins..."

# Ensure target directory exists
mkdir -p "$INSTALL_DIR/kortex"

# Copy plugin.yaml
cp "$REPO_DIR/plugin.yaml" "$INSTALL_DIR/plugin.yaml"

# Copy all .py files from kortex/ to kortex/kortex/
cp "$REPO_DIR/kortex/*.py" "$INSTALL_DIR/kortex/"

# Clean up any stale __pycache__
find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "✅ Done. Installed files:"
find "$INSTALL_DIR" -type f | sort
