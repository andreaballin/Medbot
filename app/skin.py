"""Skin lesion classifier inference (CPU is fine). Demo only: never a diagnosis."""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

from . import config

# RAG search terms (class name -> query)
RAG_QUERY = {
    "actinic keratosis / intraepithelial carcinoma": "actinic keratosis",
    "basal cell carcinoma": "basal cell skin cancer",
    "benign keratosis-like lesion": "seborrheic keratosis",
    "dermatofibroma": "dermatofibroma skin nodule",
    "melanoma": "melanoma skin cancer",
    "melanocytic nevus (mole)": "moles nevus",
    "vascular lesion": "hemangioma vascular skin lesion",
}
_cache = {}


def _load():
    if "m" not in _cache:
        ck = torch.load(config.SKIN_MODEL, map_location="cpu")
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(ck["classes"]))
        m.load_state_dict(ck["state"])
        s = ck.get("size", 224)
        _cache["m"], _cache["c"] = m.eval(), ck["classes"]
        _cache["tf"] = T.Compose([T.Resize((s, s)), T.ToTensor(),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return _cache["m"], _cache["c"]


def predict(image, top=3):
    """image: path or PIL image. Returns ([(class, probability)], recognised)."""
    m, classes = _load()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    x = _cache["tf"](img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0]
    vals, idx = p.topk(top)
    res = [(classes[i], float(v)) for v, i in zip(vals, idx)]
    return res, res[0][1] >= config.SKIN_MIN_CONF


def explain_question(class_name):
    return f"What is {RAG_QUERY[class_name]}? Describe it and how it is usually evaluated."
