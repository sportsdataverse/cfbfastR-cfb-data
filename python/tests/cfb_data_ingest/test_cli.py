from cfb_data_ingest.cli import build_parser


def test_parser_has_seasons_and_cache():
    ns = build_parser().parse_args(["--seasons", "2023", "2024", "--cache-dir", "/tmp/c"])
    assert ns.seasons == [2023, 2024] and ns.cache_dir == "/tmp/c"
