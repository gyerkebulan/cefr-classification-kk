import random, os, numpy as np, torch

CEFR2ID = {"A1":0,"A2":1,"B1":2,"B2":3,"C1":4,"C2":5}
ID2CEFR = {v:k for k,v in CEFR2ID.items()}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cefr_label_to_id(label: str) -> int:
    return CEFR2ID.get(label, -1)

def cefr_id_to_label(i: int) -> str:
    return ID2CEFR.get(int(i), "Unknown")
