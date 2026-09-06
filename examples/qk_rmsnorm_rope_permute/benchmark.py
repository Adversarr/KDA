"""Verify, benchmark or check adjoints of this example's withheld golden."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _benchmark import main


if __name__ == "__main__":
    main(default_golden=Path(__file__).resolve().with_name("golden"))
