TASKS = {
    "H_pylori_infection_status": ["negative", "positive"],
    "Presence_of_atrophy": ["non_atrophic", "atrophic"],
    "Distribution_of_atrophy": ["normal", "antrum_predominant", "corpus_predominant", "antrum_and_corpus"],
    "Gastritis_type": ["normal", "type_A", "type_B"],
}

IMAGE_SIZE = 224

# Cross-validation folds whose QMT-PAT model is used. One fold keeps the app fast
# and the bundled sample patients truly held out (they are fold-0 test patients).
# Set to [0, 1, 2, 3, 4] for a 5-fold ensemble (5x slower, and each sample patient
# was then in the training set of 4 of the 5 models).
FOLDS = [0]

# Model weights are hosted on the Hugging Face Hub, not in this git repo.
HF_CHECKPOINTS_REPO = "Hrishita-P/qmt-pat-checkpoints"
