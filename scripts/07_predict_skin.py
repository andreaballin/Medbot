"""Inferenza del classificatore cutaneo (anche su CPU).

    pip install torch torchvision pillow
    python 07_predict_skin.py foto.jpg
"""
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

MIN_CONF = 0.50  # sotto questa soglia: "non riconosciuto"
# termini di ricerca per il RAG (nome classe -> query)
RAG_QUERY = {
    "actinic keratosis / intraepithelial carcinoma": "actinic keratosis",
    "basal cell carcinoma": "basal cell skin cancer",
    "benign keratosis-like lesion": "seborrheic keratosis",
    "dermatofibroma": "dermatofibroma skin nodule",
    "melanoma": "melanoma skin cancer",
    "melanocytic nevus (mole)": "moles nevus",
    "vascular lesion": "hemangioma vascular skin lesion",
}
_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_cache = {}


def _load(path="models/skin_model.pt"):
    if "m" not in _cache:
        ck = torch.load(path, map_location="cpu")
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
        m.load_state_dict(ck["state"])
        _cache["m"], _cache["c"] = m.eval(), ck["classes"]
        s = ck.get("size", 224)
        _cache["tf"] = T.Compose([T.Resize((s, s)), T.ToTensor(),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return _cache["m"], _cache["c"]


def predict(image_path, top=3):
    m, classes = _load()
    x = _cache["tf"](Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0]
    vals, idx = p.topk(top)
    res = [(classes[i], float(v)) for v, i in zip(vals, idx)]
    return res, res[0][1] >= MIN_CONF


if __name__ == "__main__":
    res, ok = predict(sys.argv[1])
    for c, p in res:
        print(f"{p:6.1%}  {c}")
    if not ok:
        print("Confidence too low: image not recognised.")
