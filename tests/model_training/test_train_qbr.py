"""xQBR trainer: served-feature join, the game-clustered gate, the choice rule, the publish check."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from cfb_model_build.model_training import train_qbr as T
from cfb_model_build.model_training.qbr_labels import parse_week

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "model_training"


def test_parse_week_reads_ids_and_qbr_from_a_real_payload():
    payload = json.loads((FIX / "qbr_week_2024_w1_sample.json").read_text())
    rows = parse_week(payload, 2024, 2, 1)
    assert [r["game_id"] for r in rows] == [401628460, 401632062, 401643776]
    assert [r["QBR"] for r in rows] == [65.613, 94.469, 85.816]
    assert all(isinstance(r["athlete_id"], int) and r["TQBR"] is not None for r in rows)


def test_parse_week_refuses_a_second_page():
    with pytest.raises(ValueError, match="pages"):
        parse_week({"pageCount": 2, "items": []}, 2024, 2, 1)


def test_gate_resamples_whole_games():
    """99 improved rows in one game, 1 worse row in another: a row-level bootstrap would
    pass (mean -9.8, tight CI); resampling the two games leaves {B, B} a quarter of the
    time, so the upper bound is the worse game's +10 and the gate must not pass."""
    y = np.zeros(100)
    inc = np.full(100, np.sqrt(10.0))
    arm = np.zeros(100)
    arm[99] = np.sqrt(20.0)
    games = np.array([1] * 99 + [2])
    g = T.paired_gate(y, arm, inc, games)
    assert g["delta_mse"] == pytest.approx(-9.8) and g["n_games"] == 2
    assert g["ci_hi"] == pytest.approx(10.0)


def test_gate_is_strict_an_identical_model_never_passes():
    rng = np.random.default_rng(1)
    y, p = rng.normal(50, 20, 600), rng.normal(50, 20, 600)
    g = T.paired_gate(y, p, p, np.arange(600) // 2)
    assert g["delta_mse"] == 0 and g["ci_hi"] == 0  # not < 0


@pytest.mark.parametrize(
    ("ns", "sp", "want"),
    [
        ((True, 14.00), (True, 13.80), "no_spread"),  # costs 0.20 <= 0.30
        ((True, 14.00), (True, 13.50), "spread"),  # costs 0.50 > 0.30
        ((True, 99.00), (False, 1.00), "no_spread"),  # spread failed its gate
        ((False, 1.00), (True, 99.00), "spread"),
        ((False, 1.00), (False, 1.00), None),
    ],
)
def test_choice_rule(ns, sp, want):
    res = {
        "no_spread": {"passed": ns[0], "rmse": ns[1]},
        "spread": {"passed": sp[0], "rmse": sp[1]},
    }
    assert T.choose(res) == want


def _booster(path: Path) -> Path:
    X = np.random.default_rng(0).random((50, 2))
    xgb.train(
        {"objective": "reg:squarederror"}, xgb.DMatrix(X, label=X[:, 0]), 2
    ).save_model(str(path))
    return path


def test_check_gate_refuses_missing_failed_and_swapped_models(tmp_path):
    m = _booster(tmp_path / "qbr.ubj")
    with pytest.raises(RuntimeError, match="no gate record"):
        T.check_gate(m)
    rec = tmp_path / "qbr.gate.json"
    rec.write_text(json.dumps({"passed": False, "candidate_sha256": T.sha256(m)}))
    with pytest.raises(RuntimeError, match="passed=False"):
        T.check_gate(m)
    rec.write_text(json.dumps({"passed": True, "candidate_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="sha256"):
        T.check_gate(m)
    rec.write_text(json.dumps({"passed": True, "candidate_sha256": T.sha256(m)}))
    assert T.check_gate(m)["passed"] is True


def _write_season(root: Path, season: int, games: list[int], rng) -> None:
    """One adv_passing + pbp parquet pair: two passers per game, ids recoverable from pbp."""
    adv, pbp = [], []
    for g in games:
        for t in (1, 2):
            name, pid = f"qb{g}_{t}", g * 10 + t
            feats = {f: float(rng.normal()) for f in T.SERVED[:6]}
            adv.append(
                {
                    "game_id": g,
                    "pos_team_id": t,
                    "passer_player_name": name,
                    "season": season,
                    **feats,
                    "era0": 0,
                    "era1": 0,
                    "era2": 0,
                    "era3": 1,
                    "exp_qbr": 50.0,
                }
            )
            pbp += [
                {
                    "game_id": g,
                    "pos_team_id": t,
                    "passer_player_name": name,
                    "passer_player_id": pid,
                }
            ] * 3
    for ds, rows, stem in (
        ("adv_passing", adv, "adv_passing"),
        ("pbp", pbp, "play_by_play"),
    ):
        d = root / ds / "parquet"
        d.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(d / f"{stem}_{season}.parquet")


def test_served_features_takes_the_id_mode_and_drops_ambiguous_rows(tmp_path):
    _write_season(tmp_path, 2025, [7], np.random.default_rng(0))
    p = tmp_path / "pbp" / "parquet" / "play_by_play_2025.parquet"
    pbp = pl.read_parquet(p)
    # a stray wrong id on one play: the mode still wins
    pbp = pl.concat(
        [pbp, pbp.head(1).with_columns(passer_player_id=pl.lit(999, pl.Int64))]
    )
    # a second spelling of qb7_1 on team 1 maps to the same athlete -> ambiguous, both dropped
    adv = pl.read_parquet(
        tmp_path / "adv_passing" / "parquet" / "adv_passing_2025.parquet"
    )
    adv = pl.concat(
        [adv, adv.head(1).with_columns(passer_player_name=pl.lit("Q. B. Seven"))]
    )
    pbp = pl.concat(
        [pbp, pbp.head(1).with_columns(passer_player_name=pl.lit("Q. B. Seven"))]
    )
    adv.write_parquet(tmp_path / "adv_passing" / "parquet" / "adv_passing_2025.parquet")
    pbp.write_parquet(p)
    out = T.served_features(tmp_path, [2025])
    assert out["athlete_id"].to_list() == [72]
    assert out.schema["athlete_id"] == pl.Int64


def _toy_corpus(tmp_path, monkeypatch):
    """Two train seasons + a 2026 holdout of 300 games (600 QB-games >= MIN_HOLDOUT)."""
    rng = np.random.default_rng(3)
    monkeypatch.setattr(T, "TRAIN_SEASONS", (2024, 2025))
    seasons = {
        2024: list(range(1000, 1400)),
        2025: list(range(2000, 2400)),
        2026: list(range(3000, 3300)),
    }
    for s, g in seasons.items():
        _write_season(tmp_path, s, g, rng)
    feats = T.served_features(tmp_path, list(seasons))
    # the label is a smooth function of the served features, so a retrain learns it
    lab = (
        pl.DataFrame(
            [
                {
                    "season": s,
                    "season_type": 2,
                    "week": 1 + i % 4,
                    "game_id": g,
                    "athlete_id": g * 10 + t,
                }
                for s, games in seasons.items()
                for i, g in enumerate(games)
                for t in (1, 2)
            ]
        )
        .join(
            feats.select("game_id", "athlete_id", "qbr_epa", "rush_epa"),
            on=["game_id", "athlete_id"],
        )
        .with_columns(
            QBR=50 + 20 * pl.col("qbr_epa") + 10 * pl.col("rush_epa"),
            TQBR=50 + 20 * pl.col("qbr_epa"),
        )
    )
    lp = tmp_path / "labels.parquet"
    lab.drop("qbr_epa", "rush_epa").write_parquet(lp)
    return lp


def test_train_qbr_ships_a_passing_arm_with_its_record(tmp_path, monkeypatch):
    lp = _toy_corpus(tmp_path, monkeypatch)
    # an incumbent that predicts a constant: any retrain on the served features beats it
    X = pl.DataFrame({f: np.zeros(10) for f in T.SERVED}).to_pandas()
    inc = tmp_path / "flat.ubj"
    xgb.train(
        {"objective": "reg:squarederror"}, xgb.DMatrix(X, label=np.zeros(10)), 1
    ).save_model(str(inc))
    out = tmp_path / "art" / "qbr.ubj"
    assert T.train_qbr(tmp_path, lp, inc, out) == 0
    rec = T.check_gate(out)
    assert (
        rec["chosen"] == "no_spread"
    )  # spread carries no signal here, so it cannot cost > 0.30
    assert (
        rec["holdout"]["holdout_matched"] == 600 and rec["holdout"]["match_rate"] == 1.0
    )
    part = pl.read_parquet(out.parent / "qbr_partition.parquet")
    assert set(part.filter(pl.col("split") == "holdout")["season"]) == {2026}
    assert set(part.filter(pl.col("split") == "train")["season"]) == {2024, 2025}
    m = xgb.Booster()
    m.load_model(str(out))
    assert "spread" not in m.feature_names and "rush_epa" in m.feature_names
    assert (out.parent / "loso_qbr_oof.parquet").exists()


def test_train_qbr_writes_no_model_when_nothing_beats_the_incumbent(tmp_path, monkeypatch):
    lp = _toy_corpus(tmp_path, monkeypatch)
    # an incumbent fitted on train AND holdout has seen the answers: no arm beats it
    feats = T.served_features(tmp_path, [2024, 2025, 2026])
    frame, _ = T.labelled(feats, pl.read_parquet(lp))
    inc = tmp_path / "inc.ubj"
    T.fit(frame, T.ARMS["no_spread"]).save_model(str(inc))
    out = tmp_path / "art" / "qbr.ubj"
    out.parent.mkdir()
    out.write_bytes(b"a previous run's model")  # must not survive a failing run
    assert T.train_qbr(tmp_path, lp, inc, out) == T.GATE_EXIT
    assert not out.exists()
    rec = json.loads(out.with_suffix(".gate.json").read_text())
    assert rec["passed"] is False and rec["chosen"] is None
    assert rec["arms"]["no_spread"]["ci_hi"] > 0 and rec["arms"]["spread"]["ci_hi"] > 0


def test_an_arm_identical_to_the_incumbent_does_not_pass(tmp_path, monkeypatch):
    """Retraining the same recipe on the same rows reproduces the incumbent bit for bit;
    a tie is not a win, so a re-run with nothing new can never publish."""
    lp = _toy_corpus(tmp_path, monkeypatch)
    feats = T.served_features(tmp_path, [2024, 2025, 2026])
    frame, stats = T.labelled(feats, pl.read_parquet(lp))
    inc = T.fit(frame.filter(pl.col("split") == "train"), T.ARMS["no_spread"])
    results, _ = T.evaluate(frame, stats, inc)
    assert results["no_spread"]["delta_mse"] == 0 and results["no_spread"]["passed"] is False
