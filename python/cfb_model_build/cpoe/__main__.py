"""Allow ``python -m cfb_model_build.cpoe`` to invoke the CLI."""
from __future__ import annotations

import sys
from cfb_model_build.cpoe.cli import main

sys.exit(main())
