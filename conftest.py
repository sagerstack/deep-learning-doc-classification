import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pre-import src package so downstream app.src.* imports that use
# `from src.graph import ...` resolve correctly under pytest.
import src  # noqa: F401
