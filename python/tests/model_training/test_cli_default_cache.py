from model_training.cli import build_parser


def test_train_ep_final_dir_defaults_to_cache():
    ns = build_parser().parse_args(["train-ep", "--seasons", "2024", "--out", "ep.ubj"])
    assert ".cache/cfb_final" in ns.final_dir.replace("\\", "/")
