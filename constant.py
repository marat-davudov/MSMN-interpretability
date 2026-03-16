import os

DATA_DIR = "./sample_data/"
MIMIC_2_DIR = os.path.join(DATA_DIR, "mimic2")
MIMIC_3_DIR = os.path.join(DATA_DIR, "mimic3")
CHECKPOINTS_DIR = "./checkpoints"
EMBEDDING_DIR = "./embedding"
EMBEDDING_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "embedding", "word2vec_sg0_100.model")
MSMN_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "MSMN", "mimic3-50.pth")
IG_STEPS = 100
THRESHOLD = 0.5 # arbitralily chosen for demonstration purposes, should be tuned