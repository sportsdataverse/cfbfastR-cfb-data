"""Allow ``python -m cfb_model_build.cfb_matchup`` to invoke the CLI."""

from __future__ import annotations

import sys

from cfb_model_build.cfb_matchup.cli import main

sys.exit(main())
