"""Tests for cfb_model_reports.narratives — the authored prose blocks."""
from cfb_model_reports.narratives import NARRATIVES, ModelNarrative


def test_all_expected_models_have_narratives():
    for mt in ("ep", "wp_spread", "wp_naive", "qbr", "fourth_down", "rb_eval",
               "fg", "xpass", "two_pt"):
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
    # New heads: fg / xpass / two_pt load-bearing facts.
    assert "42,589" in NARRATIVES["fg"].recipe
    assert "0.0085" in NARRATIVES["fg"].recipe
    assert "1.9M" in NARRATIVES["xpass"].recipe
    assert "pass_oe" in NARRATIVES["xpass"].summary
    assert "830" in NARRATIVES["xpass"].importance  # down dominates by gain
    assert "1,622 attempts" in NARRATIVES["two_pt"].recipe
    assert "0.9851" in NARRATIVES["two_pt"].summary  # XP make rate


def test_new_heads_have_methodology_sections():
    """fg / xpass / two_pt carry the full enriched methodology blocks."""
    for mt in ("fg", "xpass", "two_pt"):
        n = NARRATIVES[mt]
        assert n.features.strip() and "| Feature | Type |" in n.features, mt
        assert "XGBoost" in n.model and "binary:logistic" in n.model, mt
        assert n.importance.strip(), mt


def test_methodology_sections_present_for_core_models():
    """Each narrative carries the enriched methodology blocks: a per-feature
    'Model features' table, a 'The model' algorithm/CV block, and a 'Feature
    importance' note — the nflfastR-post structure.
    """
    for mt in ("ep", "wp_spread", "wp_naive", "qbr", "fourth_down", "rb_eval"):
        n = NARRATIVES[mt]
        assert n.features.strip(), f"{mt}.features empty"
        assert n.model.strip(), f"{mt}.model empty"
        assert n.importance.strip(), f"{mt}.importance empty"


def test_features_blocks_describe_each_feature_as_table():
    """The 'Model features' block is a Markdown table (one row per feature)."""
    for mt in ("ep", "wp_spread", "qbr", "fourth_down", "rb_eval"):
        feats = NARRATIVES[mt].features
        assert "| Feature | Type |" in feats, f"{mt} features not a table"


def test_model_blocks_name_algorithm_and_cv():
    """'The model' must name the algorithm and the leave-one-season-out CV."""
    for mt in ("ep", "wp_spread", "wp_naive", "qbr"):
        model = NARRATIVES[mt].model
        assert "XGBoost" in model, f"{mt} model missing algorithm"
        assert "eave-one-season-out" in model or "LOSO" in model, f"{mt} model missing CV"
    assert "525 boosting rounds" in NARRATIVES["ep"].model
    assert "760 boosting rounds" in NARRATIVES["wp_spread"].model
