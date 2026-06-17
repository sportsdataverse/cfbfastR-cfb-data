import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def packages_dir() -> pathlib.Path:
    # python/tests/conftest.py -> parents[1] == python/ (the packages root)
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def final_cache_dir() -> pathlib.Path:
    # Where cfb_data_ingest caches final.json. Gitignored. Overridable for integration runs.
    env = os.environ.get("CFB_FINAL_CACHE")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[1] / ".cache" / "cfb_final"
