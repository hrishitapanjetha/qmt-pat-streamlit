"""Patient-level inference with QMT-PAT.

For each fold in config.FOLDS:
  images -> fine-tuned ViT-B/16 (768-d feature per image)
         -> frozen patient-attention anchor (one summary + logits per task)
         -> 16-qubit task-structured circuit -> residual correction delta
         -> final logits = anchor logits + delta
Probabilities are averaged across folds when more than one fold is used.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from config import FOLDS, HF_CHECKPOINTS_REPO, TASKS
from models import MAX_IMAGES_EVAL, PatientAttentionAnchor, QMT_PAT_V9, ViTMultiTask
from transforms import eval_transform

APP_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = APP_DIR / "checkpoints"
SAMPLES_DIR = APP_DIR / "samples"
TASK_ORDER = list(TASKS)

VIT_CKPT = "vit_b16_fold{fold}_best.pt"
QMTPAT_CKPT = "v9_quantum_fold{fold}_best.pt"

CLASS_LABELS = {
    "H_pylori_infection_status": ["Negative", "Positive"],
    "Presence_of_atrophy": ["Non-atrophic", "Atrophic"],
    "Distribution_of_atrophy": ["Normal", "Antrum-predominant", "Corpus-predominant", "Antrum & Corpus"],
    "Gastritis_type": ["Normal", "Type A (Autoimmune)", "Type B (H. pylori)"],
}

TASK_TITLES = {
    "H_pylori_infection_status": "H. pylori Infection Status",
    "Presence_of_atrophy": "Presence of Atrophy",
    "Distribution_of_atrophy": "Distribution of Atrophy",
    "Gastritis_type": "Gastritis Type",
}

# Deterministic EGID label hierarchy: atrophy distribution fixes the other two atrophy labels.
HIERARCHY = {
    "normal": ("non_atrophic", "normal"),
    "antrum_predominant": ("atrophic", "type_B"),
    "corpus_predominant": ("atrophic", "type_A"),
    "antrum_and_corpus": ("atrophic", "type_B"),
}


def label_of(task: str, raw: str) -> str:
    return CLASS_LABELS[task][TASKS[task].index(raw)]


def get_device():
    # CPU: the PennyLane circuit is simulated on CPU, and HF Spaces ZeroGPU only
    # grants CUDA inside @spaces.GPU functions, which this app doesn't use.
    return torch.device("cpu")


def _resolve_checkpoint(filename: str) -> Path:
    """Prefer a local checkpoints/ copy if present; otherwise download from the
    Hugging Face model repo (cached by huggingface_hub after the first download)."""
    local_path = CHECKPOINTS_DIR / filename
    if local_path.exists():
        return local_path
    return Path(hf_hub_download(repo_id=HF_CHECKPOINTS_REPO, filename=filename))


def _state_dict(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("model_state_dict", payload)


@lru_cache(maxsize=1)
def load_models():
    device = get_device()
    models = []
    for fold in FOLDS:
        vit = ViTMultiTask()
        vit.load_state_dict(_state_dict(_resolve_checkpoint(VIT_CKPT.format(fold=fold))))
        vit.to(device).eval()

        qmt = QMT_PAT_V9(PatientAttentionAnchor(), core_type="quantum")
        qmt.load_state_dict(_state_dict(_resolve_checkpoint(QMTPAT_CKPT.format(fold=fold))))
        qmt.to(device).eval()
        models.append((fold, vit, qmt))
    return models, device


def load_samples() -> dict:
    path = SAMPLES_DIR / "samples.json"
    return json.loads(path.read_text()) if path.exists() else {}


@torch.no_grad()
def predict(images: list[Image.Image]):
    """Patient-level prediction from all of one patient's images."""
    models, device = load_models()
    n_images = len(images)
    if n_images > MAX_IMAGES_EVAL:  # evenly spaced, as in the paper's evaluation
        idx = np.linspace(0, n_images - 1, MAX_IMAGES_EVAL).round().astype(int)
        images = [images[i] for i in idx]
    xs = torch.stack([eval_transform(img.convert("RGB")) for img in images]).to(device)
    mask = torch.zeros(1, xs.shape[0], dtype=torch.bool, device=device)

    acc = {k: {t: [] for t in TASK_ORDER} for k in ("final", "anchor", "vit", "delta")}
    for _fold, vit, qmt in models:
        feats = torch.cat([vit.forward_features(xs[i:i + 16]) for i in range(0, len(xs), 16)])
        vit_logits = {t: h(feats) for t, h in vit.heads.items()}
        out = qmt(feats.unsqueeze(0), mask)
        for ti, task in enumerate(TASK_ORDER):
            acc["final"][task].append(torch.softmax(out["final_logits"][ti], 1)[0].numpy())
            acc["anchor"][task].append(torch.softmax(out["anchor_logits"][ti], 1)[0].numpy())
            acc["vit"][task].append(torch.softmax(vit_logits[task], 1).mean(0).numpy())
            acc["delta"][task].append(out["deltas"][ti][0].numpy())

    results = {}
    for task in TASK_ORDER:
        probs = np.mean(acc["final"][task], axis=0)
        pred_idx = int(probs.argmax())
        fold_votes = [int(p.argmax()) for p in acc["final"][task]]
        results[task] = {
            "predicted_raw": TASKS[task][pred_idx],
            "predicted_class": CLASS_LABELS[task][pred_idx],
            "predicted_idx": pred_idx,
            "confidence": float(probs[pred_idx]),
            "all_probs": {CLASS_LABELS[task][i]: float(p) for i, p in enumerate(probs)},
            "anchor_probs": {CLASS_LABELS[task][i]: float(p) for i, p in enumerate(np.mean(acc["anchor"][task], axis=0))},
            "vit_probs": {CLASS_LABELS[task][i]: float(p) for i, p in enumerate(np.mean(acc["vit"][task], axis=0))},
            "delta": np.mean(acc["delta"][task], axis=0).tolist(),
            "model_agreement": sum(v == pred_idx for v in fold_votes) / len(fold_votes),
        }

    dist = results["Distribution_of_atrophy"]["predicted_raw"]
    exp_atrophy, exp_type = HIERARCHY[dist]
    hierarchy = {
        "distribution": label_of("Distribution_of_atrophy", dist),
        "expected_atrophy": label_of("Presence_of_atrophy", exp_atrophy),
        "expected_type": label_of("Gastritis_type", exp_type),
        "consistent": results["Presence_of_atrophy"]["predicted_raw"] == exp_atrophy
        and results["Gastritis_type"]["predicted_raw"] == exp_type,
    }
    return results, hierarchy, len(models), n_images
