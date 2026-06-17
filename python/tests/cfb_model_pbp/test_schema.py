from cfb_model_pbp.schema import MODEL_PBP_COLUMNS, JOIN_KEYS, CARRY_RENAME


def test_schema_contract():
    assert JOIN_KEYS == ("game_id", "id")
    for c in ("game_id", "id", "epa", "wpa", "cpoe", "completion_prob", "ep_before", "wp_after"):
        assert c in MODEL_PBP_COLUMNS
    # EP/WP are carried (renamed) from final.json, not re-scored in SP1
    assert CARRY_RENAME["EPA"] == "epa" and CARRY_RENAME["wp_before"] == "wp_before"
