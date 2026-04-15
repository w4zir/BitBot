"""Pytest configuration."""

from __future__ import annotations

import os

# Ensure classifier env exists for app import in tests that don't mock it early.
os.environ.setdefault("CLASSIFIER_BENTOML_URL", "http://127.0.0.1:9/classify")
