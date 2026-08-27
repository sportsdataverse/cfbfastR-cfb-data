"""CFB modelling pipeline: ingest -> features -> train -> gate -> package -> publish.

One package per model family, mirroring ``cfb_data_build``'s shape on the dataset
side. The numbered ``python/cfb_model_*_creation.py`` shims are the entry points;
each forwards to one of these submodules.

Members are separate MODELS, not stages of one model -- ``cpoe``, ``pregame_wp``
and ``rb_eval`` each own their features, training and gates. ``model_training``
holds the shared bundle (ep / wp / fg / qbr / xpass / two_pt / punt),
``cfb_model_pbp`` scores the corpus they all read, ``cfb_model_publish`` uploads,
and ``cfb_model_reports`` writes the cards.

Every published artifact needs a row in the Model registry in CLAUDE.md;
``tests/test_model_registry.py`` enforces that each row's fitting script resolves
to a real stage.
"""
