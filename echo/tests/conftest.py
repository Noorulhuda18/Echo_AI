"""Shared fixtures. Every test works inside a temporary folder, never in real data."""

from __future__ import annotations

import pytest

import config


@pytest.fixture
def paths(tmp_path):
    p = config.Paths.from_root(tmp_path)
    p.ensure()
    return p
