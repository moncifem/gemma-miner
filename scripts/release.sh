#!/usr/bin/env bash
# Build + publish gemma-miner to PyPI.
#
# Reads PYPI_TOKEN from .env (gitignored). Pass --test for TestPyPI.
#
# Usage:
#   scripts/release.sh           # publishes to https://pypi.org
#   scripts/release.sh --test    # publishes to https://test.pypi.org
#
# Every release should already have:
#   1. Version bumped in pyproject.toml + src/gemma_miner/__init__.py
#   2. CHANGELOG.md entry for the new version
#   3. git commit + tag pushed
set -eu

cd "$(dirname "$0")/.."

# Load .env into the environment.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

if [ -z "${PYPI_TOKEN:-}" ]; then
    echo "PYPI_TOKEN is not set. Add it to .env:"
    echo "  echo 'PYPI_TOKEN=pypi-...' >> .env"
    exit 1
fi

PUBLISH_URL="https://upload.pypi.org/legacy/"
TARGET="PyPI"
if [ "${1:-}" = "--test" ]; then
    PUBLISH_URL="https://test.pypi.org/legacy/"
    TARGET="TestPyPI"
fi

echo "building wheel + sdist..."
rm -rf dist/
uv build

echo "publishing to ${TARGET}..."
uv publish --publish-url "$PUBLISH_URL" --token "$PYPI_TOKEN"

if [ "$TARGET" = "TestPyPI" ]; then
    echo "verify at https://test.pypi.org/project/gemma-miner/"
else
    echo "verify at https://pypi.org/project/gemma-miner/"
fi
