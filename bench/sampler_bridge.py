import sys
from pathlib import Path

# Add p-less/ to sys.path so we can import without modifying it
_pless_dir = str(Path(__file__).resolve().parent.parent / "p-less")
if _pless_dir not in sys.path:
    sys.path.insert(0, _pless_dir)

from p_less_samplers import p_less_decode, p_less_norm_decode  # noqa: E402

SAMPLERS = {
    "pless": p_less_decode,
    "pless_norm": p_less_norm_decode,
}
