import sys
from pathlib import Path

# Make the repo root importable so `from axes... import ...` works under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
