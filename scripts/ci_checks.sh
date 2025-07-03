#!/usr/bin/env bash
set -e

echo "🧹 Running lint (black + isort)..."
black --check .
isort --check-only .

echo "🐍 Running type checks (mypy)..."
mypy core app training scripts tests

echo "🧪 Running tests (pytest)..."
pytest --maxfail=1 --disable-warnings -q

echo "✅ CI checks passed!"
