"""Allena un classificatore di lesioni cutanee (7 classi) su DermaMNIST-224 (derivato da HAM10000).

Pensato per Colab/Kaggle con GPU (T4):
    !pip -q install torch torchvision numpy
    !python 06_train_skin.py --epochs 8
Output: models/skin_model.pt (pesi + nomi classi), models/skin_metrics.json
Licenza dati: CC BY-NC 4.0 (solo uso non commerciale).
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

URL = "https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1"
CLASSES = ["actinic keratosis / intraepithelial carcinoma", "basal cell carcinoma",
           "benign keratosis-like lesion", "dermatofibroma", "melanoma",
           "melanocytic nevus (mole)", "vascular lesion"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class DS(Dataset):
    def __init__(self, x, y, tf):
        self.x, self.y, self.tf = x, y.reshape(-1), tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), int(self.y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--size", type=int, default=224, help="risoluzione input (128 per CPU)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    f = Path("data/dermamnist_224.npz")
    Path("models").mkdir(exist_ok=True)
    if not f.exists():
        print("scarico dataset (1 GB)...")
        urllib.request.urlretrieve(URL, f)
    d = np.load(f)
    train_tf = T.Compose([T.ToPILImage(), T.Resize((a.size, a.size)), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToPILImage(), T.Resize((a.size, a.size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = DataLoader(DS(d["train_images"], d["train_labels"], train_tf), a.bs, shuffle=True, num_workers=2)
    va = DataLoader(DS(d["val_images"], d["val_labels"], eval_tf), a.bs, num_workers=2)
    te = DataLoader(DS(d["test_images"], d["test_labels"], eval_tf), a.bs, num_workers=2)

    counts = np.bincount(d["train_labels"].reshape(-1), minlength=7)
    w = torch.tensor((counts.sum() / (7 * counts)) ** 0.5, dtype=torch.float32, device=dev)  # pesi attenuati
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.epochs * len(tr))
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    def evaluate(loader):
        m.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y in loader:
                P.append(m(x.to(dev)).argmax(1).cpu())
                Y.append(y)
        P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        recall = [float((P[Y == c] == c).mean()) if (Y == c).any() else 0.0 for c in range(7)]
        return float((P == Y).mean()), float(np.mean(recall)), recall

    best = 0
    for ep in range(a.epochs):
        m.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(m(x), y)
            loss.backward()
            opt.step()
            sched.step()
        acc, bacc, _ = evaluate(va)
        print(f"epoch {ep+1}: val acc {acc:.3f}  balanced acc {bacc:.3f}  loss {loss.item():.3f}")
        if bacc > best:
            best = bacc
            torch.save({"state": m.state_dict(), "classes": CLASSES, "size": a.size}, "models/skin_model.pt")

    ck = torch.load("models/skin_model.pt", map_location=dev)
    m.load_state_dict(ck["state"])
    acc, bacc, recall = evaluate(te)
    metrics = {"test_acc": acc, "test_balanced_acc": bacc,
               "recall_per_class": dict(zip(CLASSES, recall))}
    json.dump(metrics, open("models/skin_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
