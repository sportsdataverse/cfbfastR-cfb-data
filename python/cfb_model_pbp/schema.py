from __future__ import annotations

JOIN_KEYS = ("game_id", "id")

IDENTITY_COLS = ["game_id", "id", "sequenceNumber", "game_play_number", "drive.id", "season", "week", "period"]

DESCRIPTOR_COLS = [
    "pos_team", "def_pos_team", "start.pos_team.name", "homeTeamId", "awayTeamId",
    "homeTeamName", "awayTeamName", "type.text", "text", "start.down", "start.distance",
    "start.yardsToEndzone", "pos_score_diff_start", "start.TimeSecsRem", "start.is_home",
    "passing_down", "pass", "rush", "completion", "scoring_play", "statYardage", "passer_player_name",
]

PREDICTION_COLS = [
    "ep_before", "ep_after", "epa", "def_epa",
    "wp_before", "wp_after", "wpa", "def_wp_before", "def_wp_after",
    "home_wp_before", "away_wp_before", "home_wp_after", "away_wp_after",
    "completion_prob", "cpoe",
    "model_pbp_version", "ep_model_version", "wp_model_version", "cp_model_version", "scored_date",
]

MODEL_PBP_COLUMNS = IDENTITY_COLS + DESCRIPTOR_COLS + PREDICTION_COLS

# EP/WP/EPA/WPA are CARRIED from final.json (they already embed CFBPlayProcess differencing);
# only CPOE is net-new scored in SP1. Map final.json source names -> frozen snake names.
CARRY_RENAME = {
    "EP_start": "ep_before", "EP_end": "ep_after", "EPA": "epa",
    "wp_before": "wp_before", "wp_after": "wp_after", "wpa": "wpa",
}
