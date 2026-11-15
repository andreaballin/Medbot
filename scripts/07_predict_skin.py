"""Skin classifier inference from the terminal (CPU is fine). Demo only: never a diagnosis.

    pip install torch torchvision pillow
    python scripts/07_predict_skin.py photo.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.skin import predict  # noqa: E402

if __name__ == "__main__":
    res, ok = predict(sys.argv[1])
    for c, p in res:
        print(f"{p:6.1%}  {c}")
    if not ok:
        print("Confidence too low: image not recognised.")
