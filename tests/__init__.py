"""Define basic paths needed for testing."""
from pathlib import Path

_my_dir = Path(__file__).resolve().parent
config_dir = _my_dir / "configs"
backward_compatibility_dir = _my_dir / "backward_compatibility"
examples_dir = _my_dir.parent / "examples"
output_dir = _my_dir / "output"
other_dir = _my_dir / "other"
train_dir = _my_dir / "train"
test_dir = _my_dir / "test"
