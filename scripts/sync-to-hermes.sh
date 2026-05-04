#!/bin/bash
# Sync hermes-kortex from repo to ~/.hermes/plugins/kortex
#
# IMPORTANT: The installed plugin must be FLAT:
#   ~/.hermes/plugins/kortex/plugin.yaml
#   ~/.hermes/plugins/kortex/__init__.py
#   ~/.hermes/plugins/kortex/provider.py
#   ...
#
# The memory provider and context engine discovery systems look for
# __init__.py DIRECTLY inside ~/.hermes/plugins/kortex/ — NOT in a nested
# subdirectory. If you nest it (kortex/kortex/__init__.py), both systems
# fail to find it.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="$HOME/.hermes/plugins/kortex"

echo "📦 Syncing hermes-kortex to Hermes plugins..."

# Remove old installation (clean slate)
rm -rf "$INSTALL_DIR"

# Create FLAT target directory
mkdir -p "$INSTALL_DIR"

# Copy plugin.yaml
cp "$REPO_DIR/plugin.yaml" "$INSTALL_DIR/"

# Copy all .py files FLAT (NOT into a nested kortex/ subdir)
cp "$REPO_DIR/kortex/*.py" "$INSTALL_DIR/"

# Clean up any stale __pycache__
find "$INSTALL_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "✅ Done. Installed files:"
find "$INSTALL_DIR" -type f | sort
