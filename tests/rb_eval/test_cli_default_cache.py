from rb_eval.cli import build_parser


def test_features_final_dir_defaults_to_cache():
    ns = build_parser().parse_args(["features"])
    assert ".cache/cfb_final" in ns.final_dir.replace("\\", "/")
