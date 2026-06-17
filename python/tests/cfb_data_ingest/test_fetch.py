import json
import polars as pl
from cfb_data_ingest.fetch import fetch_final


def _fake_downloader(url, **kwargs):
    gid = url.rsplit("/", 1)[-1].removesuffix(".json")
    class R:  # mimic requests.Response surface used by fetch_final
        status_code = 200
        text = json.dumps({"season": 2024, "plays": [{"id": int(gid)}]})
    return R()


def test_fetch_final_writes_cache_and_counts(tmp_path):
    sched = tmp_path / "sched.parquet"
    pl.DataFrame({"game_id": [10, 11], "season": [2024, 2024]}).write_parquet(sched)
    cache = tmp_path / "cache"
    stats = fetch_final([2024], cache, schedule=sched, downloader=_fake_downloader)
    assert stats == {"fetched": 2, "skipped": 0, "missing": 0, "total": 2}
    assert (cache / "10.json").exists() and (cache / "11.json").exists()
    # second run skips cached
    stats2 = fetch_final([2024], cache, schedule=sched, downloader=_fake_downloader)
    assert stats2["skipped"] == 2 and stats2["fetched"] == 0
