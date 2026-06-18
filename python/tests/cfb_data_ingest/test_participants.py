"""Hermetic tests for the pre-2014 participant name+id resolver."""
from __future__ import annotations

import copy

from cfb_data_ingest.participants import fill_participants_from_text


def _box(cat, name, athlete_id):
    return {"name": cat, "athletes": [{"athlete": {"displayName": name, "id": athlete_id}}]}


# One team's boxscore covering every recoverable category.
RAW = {
    "boxscore": {
        "players": [
            {
                "statistics": [
                    _box("rushing", "Mike Kafka", "169280"),
                    _box("passing", "Chris Todd", "111"),
                    _box("receiving", "Mario Fannin", "222"),
                    _box("interceptions", "Walter McFadden", "333"),
                    _box("kicking", "Wes Byrum", "444"),
                    _box("punting", "Stefan Demos", "555"),
                    _box("kickReturns", "Stephen Simmons", "666"),
                    _box("puntReturns", "Greg Reid", "777"),
                ]
            }
        ]
    }
}

# Each play exercises one role; all participant columns start null (pre-2014 shape).
PLAYS = [
    {"text": "Mike Kafka rush for 4 yards to the Nwest 33."},
    {"text": "Chris Todd pass complete to Mario Fannin for 8 yards."},
    {"text": "Chris Todd pass intercepted by Walter McFadden at the Nwest 31."},
    {"text": "Wes Byrum 21 yard field goal GOOD."},
    {"text": "Stefan Demos punt for 43 yards, downed at the Aub 21."},
    {"text": "Wes Byrum kickoff for 64 yards returned by Stephen Simmons for 12 yards to the Nwest 33."},
    {"text": "Cam Newton punt blocked, returned by Greg Reid for 20 yards on the punt return."},
]


def _run(plays):
    p = copy.deepcopy(plays)
    res = fill_participants_from_text(p, RAW)
    return p, res


def test_rusher_name_and_id():
    p, _ = _run(PLAYS)
    assert p[0]["rusher_player_name"] == "Mike Kafka"
    assert p[0]["rusher_player_id"] == "169280"


def test_passer_and_receiver_on_completion():
    p, _ = _run(PLAYS)
    assert p[1]["passer_player_name"] == "Chris Todd"
    assert p[1]["passer_player_id"] == "111"
    assert p[1]["receiver_player_name"] == "Mario Fannin"
    assert p[1]["receiver_player_id"] == "222"


def test_interception_name_and_id():
    p, _ = _run(PLAYS)
    assert p[2]["interception_player_name"] == "Walter McFadden"
    assert p[2]["interception_player_id"] == "333"


def test_field_goal_kicker():
    p, _ = _run(PLAYS)
    assert p[3]["fg_kicker_player_name"] == "Wes Byrum"
    assert p[3]["fg_kicker_player_id"] == "444"


def test_punter():
    p, _ = _run(PLAYS)
    assert p[4]["punter_player_name"] == "Stefan Demos"
    assert p[4]["punter_player_id"] == "555"


def test_kickoff_and_kick_returner():
    p, _ = _run(PLAYS)
    assert p[5]["kickoff_player_name"] == "Wes Byrum"
    assert p[5]["kickoff_player_id"] == "444"
    assert p[5]["kickoff_return_player_name"] == "Stephen Simmons"
    assert p[5]["kickoff_return_player_id"] == "666"


def test_punt_returner_disambiguated_from_kickoff():
    p, _ = _run(PLAYS)
    assert p[6]["punt_return_player_name"] == "Greg Reid"
    assert p[6]["punt_return_player_id"] == "777"
    assert "kickoff_return_player_name" not in p[6] or p[6].get("kickoff_return_player_name") is None


def test_id_paired_for_preexisting_name():
    # CFBPlayProcess shape pre-2014: name present, id null -> id must be paired.
    plays = [{"text": "Chris Todd pass complete to Mario Fannin for 8 yards.",
              "passer_player_name": "Chris Todd"}]
    p, res = _run(plays)
    assert p[0]["passer_player_id"] == "111"
    assert "passer_player_id" in res["ids"]


def test_existing_values_not_overwritten():
    plays = [{"text": "Mike Kafka rush for 4 yards.",
              "rusher_player_name": "Someone Else", "rusher_player_id": "999"}]
    p, _ = _run(plays)
    assert p[0]["rusher_player_name"] == "Someone Else"
    assert p[0]["rusher_player_id"] == "999"


def test_regex_fallback_name_without_box_gets_null_id():
    # rusher not in the boxscore -> name via regex fallback, id stays null.
    plays = [{"text": "Unknown Backup rush for 2 yards."}]
    p, _ = _run(plays)
    assert p[0]["rusher_player_name"] == "Unknown Backup"
    assert p[0].get("rusher_player_id") is None


def test_no_boxscore_is_safe_noop():
    plays = [{"text": "Mike Kafka rush for 4 yards."}]
    p = copy.deepcopy(plays)
    res = fill_participants_from_text(p, {})
    # regex fallback still recovers the name; no boxscore -> no id
    assert p[0]["rusher_player_name"] == "Mike Kafka"
    assert p[0].get("rusher_player_id") is None
    assert res["ids"] == {}
