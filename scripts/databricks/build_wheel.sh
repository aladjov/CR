#!/bin/bash
#
# Build wheel for Databricks deployment
#
# Usage:
#   ./scripts/databricks/build_wheel.sh
#   ./scripts/databricks/build_wheel.sh --upload /Volumes/catalog/schema/packages/
#
# Output:
#   dist/customer_retention-{version}-py3-none-any.whl
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Building wheel for customer-retention..."

# Clean previous builds
rm -rf dist/ build/ src/*.egg-info

# Build wheel using uv (faster) or pip
if command -v uv &> /dev/null; then
    echo "Using uv to build..."
    uv build --wheel
else
    echo "Using pip to build..."
    python -m pip wheel . --no-deps -w dist/
fi

# Find the built wheel
WHEEL=$(ls dist/*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL" ]; then
    echo "Error: No wheel found in dist/"
    exit 1
fi

echo "Built: $WHEEL"

# Optional: Upload to Unity Catalog Volume
if [ "$1" == "--upload" ] && [ -n "$2" ]; then
    DEST="$2"
    echo "Uploading to $DEST..."

    # Check if databricks CLI is available
    if command -v databricks &> /dev/null; then
        databricks fs cp "$WHEEL" "$DEST" --overwrite
        echo "Uploaded to: ${DEST}$(basename $WHEEL)"
    else
        echo "Warning: databricks CLI not found, cannot upload"
        echo "Install with: pip install databricks-cli"
        exit 1
    fi
fi

echo "Done!"
