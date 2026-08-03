"""Sentinel: CFBD-touching ingest tests must be integration-marked.

This test verifies that ``tests.pregame_wp.test_data_ingest`` carries a
module-level ``integration`` pytestmark so the default
``-m "not integration"`` run never requires ``CFB_DATA_API_KEY``.
"""
import pytest


def test_cfbd_tests_are_integration_marked():
    # Sentinel: the live-CFBD ingest test must carry the integration marker so the
    # default `-m "not integration"` run never requires CFB_DATA_API_KEY.
    import tests.pregame_wp.test_data_ingest as t  # noqa: F401
    marks = getattr(t, "pytestmark", [])
    names = {m.name for m in (marks if isinstance(marks, list) else [marks])}
    assert "integration" in names
