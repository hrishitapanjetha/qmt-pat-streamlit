# QMT-PAT Gastritis Classifier (Streamlit)

A Streamlit app for patient-level multi-task gastritis classification from endoscopic images, running
**QMT-PAT**, the hybrid quantum-classical model from *QMT-PAT: A Controlled Evaluation of a
Task-Structured Quantum Circuit for Patient-Level Multi-Task Gastritis Classification from Endoscopic
Images*, trained on the [EGID dataset](https://doi.org/10.6084/m9.figshare.30997345).
It is the Streamlit counterpart of the Gradio app, with the same model code, samples and design.

Given all endoscopy images of one patient, the app predicts four clinical dimensions jointly:
- **H. pylori infection status** (positive / negative)
- **Presence of atrophy** (atrophic / non-atrophic)
- **Distribution of atrophy** (normal / antrum-predominant / corpus-predominant / antrum & corpus)
- **Gastritis type** (normal / Type A — autoimmune / Type B — H. pylori-associated)

## How it works

1. A fine-tuned **ViT-B/16** turns each image into a 768-d feature vector.
2. A frozen **patient-attention anchor** pools all of the patient's images into one summary per task and
   makes a first prediction.
3. A simulated **16-qubit task-structured circuit** (4 qubits per task, CZ rings within tasks, CNOTs
   across tasks; PennyLane `lightning.qubit`) adds a small residual correction to the anchor's logits.

In five-fold patient-level cross-validation QMT-PAT reached a mean balanced accuracy of 0.791
(AUROC 0.920). For each task the app shows the class probabilities before and after the quantum
correction, and flags predictions that break the EGID label hierarchy.

The **fold-0** model is used by default (`FOLDS` in `config.py`); set `FOLDS = [0, 1, 2, 3, 4]` for a
5-fold ensemble (about 5× slower; the sample patients are then no longer held out from every model).

## Sample patients

`samples/` holds four held-out **fold-0 test patients** (one per atrophy-distribution class) with their
true labels in `samples/samples.json`. QMT-PAT gets
15 of the 16 labels right (patient 5's H. pylori status is wrong). Images: EGID dataset (Renji Hospital,
Shanghai), CC BY-NC-ND 4.0, converted from BMP to JPEG (quality 95, original resolution).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Model checkpoints are not committed (see `.gitignore`). The app looks in `checkpoints/` first and
otherwise downloads them from the [Hugging Face Hub](https://huggingface.co/Hrishita-P/qmt-pat-checkpoints)
(cached after the first run):

| File | Size | Model |
|---|---|---|
| `vit_b16_fold0_best.pt` | 343 MB | fine-tuned ViT-B/16 image encoder |
| `v9_quantum_fold0_best.pt` | 4.7 MB | QMT-PAT (patient-attention anchor + 16-qubit circuit) |

## Deploying on Streamlit Community Cloud

1. Upload `vit_b16_fold0_best.pt` and `v9_quantum_fold0_best.pt` to the Hugging Face model repo named
   in `config.py` (`HF_CHECKPOINTS_REPO`). Keep it public, or add `HF_TOKEN = "hf_..."` to the app's
   Secrets for a private repo.
2. Push this folder to GitHub (checkpoints stay out of git).
3. On [share.streamlit.io](https://share.streamlit.io) create an app from the repo, with `app.py` as the
   main file and Python 3.12 under Advanced settings. The checkpoints download on first start.

The app needs about 1.4 GB of memory (Community Cloud allows up to 2.7 GB) and takes a few seconds per
patient on CPU. `requirements.txt` installs the CPU-only PyTorch build.

## Disclaimer

This is a research/educational prototype built on a public benchmark dataset. It is **not a diagnostic
device** and has not been clinically validated. Predictions should never be used to inform real patient
care or replace professional medical judgement.
