from cfb_model_pbp.cli import build_parser


def test_parser_requires_out_and_cp_model():
    ns = build_parser().parse_args(["--final-dir", ".cache/cfb_final", "--cp-model", "m.ubj", "--out", "o.parquet"])
    assert ns.out == "o.parquet" and ns.cp_model == "m.ubj"
