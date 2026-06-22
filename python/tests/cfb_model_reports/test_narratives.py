"""Tests for cfb_model_reports.narratives — the authored prose blocks."""
from cfb_model_reports.narratives import NARRATIVES, ModelNarrative


def test_all_expected_models_have_narratives():
    for mt in ("ep", "wp_spread", "wp_naive", "qbr", "fourth_down", "rb_eval"):
        assert mt in NARRATIVES, f"missing narrative for {mt}"


def test_every_narrative_has_four_nonempty_sections():
    for mt, n in NARRATIVES.items():
        assert isinstance(n, ModelNarrative)
        for field in ("summary", "recipe", "discussion", "limitations"):
            text = getattr(n, field)
            assert isinstance(text, str) and text.strip(), f"{mt}.{field} is empty"


def test_lineage_facts_present():
    """A few load-bearing lineage facts must appear verbatim in the prose."""
    assert "2,219,607" in NARRATIVES["ep"].recipe
    assert "760" in NARRATIVES["wp_spread"].recipe
    assert "65 trees" in NARRATIVES["wp_naive"].recipe
    assert "45 trees" in NARRATIVES["qbr"].recipe
    assert "76 classes" in NARRATIVES["fourth_down"].recipe
    assert "4th-most-important" in NARRATIVES["fourth_down"].recipe
    assert "897 rushers" in NARRATIVES["rb_eval"].recipe
