# gui_fusion_demo.py
# Dual-Branch Fake News Detector (BERT + TransE)
# Works with:
#   - transe_model.py (TransE, Aggregator)
#   - triplet_extraction/rebel_triplet_extractor.py (extract_triplets)
#   - models/entity_vocab.pt, models/relation_vocab.pt
#   - models/transe_model_valLoss_0.3785.pt, models/fusion_model.pt

import os
# Prevent transformers from importing TensorFlow (stops the noisy TF logs)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

from transformers import BertTokenizer, BertModel
from triplet_extraction.rebel_triplet_extractor import extract_triplets
from transe_model import TransE, Aggregator


# -------------------------
# Helpers
# -------------------------
def safe_load(path: str, device: torch.device):
    """
    Load a torch checkpoint to the given device.
    Handles both raw state_dict and {'state_dict': ...} formats.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return ckpt


# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabularies (saved as dicts: str -> int)
entity_vocab = torch.load("models/entity_vocab.pt", map_location="cpu")
relation_vocab = torch.load("models/relation_vocab.pt", map_location="cpu")

# TransE (embedding_dim must match what your fusion expects)
transe = TransE(len(entity_vocab), len(relation_vocab), embedding_dim=128)
try:
    transe.load_state_dict(safe_load("models/transe_model_valLoss_0.3785.pt", device), strict=False)
except FileNotFoundError:
    # If the checkpoint is missing, the app still runs (weights are random).
    pass
transe.to(device).eval()

# Aggregator (outputs a single knowledge vector per article)
aggregator = Aggregator(dim=transe.dim, out_dim=128).to(device).eval()

# BERT (text branch)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()


# -------------------------
# Fusion Classifier
# -------------------------
class FusionClassifier(nn.Module):
    def __init__(self, bert_dim=768, knowledge_dim=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(bert_dim + knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, cls_embedding, knowledge_vector):
        fused = torch.cat((cls_embedding, knowledge_vector), dim=1)
        return self.fusion(fused)


fusion_model = FusionClassifier(
    bert_dim=768,
    knowledge_dim=aggregator.out_dim,  # keep in sync with aggregator
    hidden_dim=256,
    num_classes=2,
).to(device)

try:
    fusion_model.load_state_dict(safe_load("models/fusion_model.pt", device), strict=False)
except FileNotFoundError:
    # If missing, app still runs (random fusion head).
    pass
fusion_model.eval()


# -------------------------
# Prediction Pipeline
# -------------------------
@torch.inference_mode()
def predict(article_text: str) -> str:
    if not article_text or not article_text.strip():
        return "📝 Please paste a news article first."

    # 1) BERT CLS embedding
    inputs = bert_tokenizer(
        article_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

    # 2) Triplet extraction + knowledge vector
    triplets = extract_triplets(article_text)
    knowledge_vector = aggregator(
        triplets=triplets,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        transe=transe,
        device=device
    )  # (1, aggregator.out_dim)

    # 3) Fusion prediction
    logits = fusion_model(cls_embedding, knowledge_vector)
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)

    label = "REAL" if pred.item() == 1 else "FAKE"
    percent = confidence.item() * 100.0
    return f"{'🟢' if label == 'REAL' else '🔴'} {label} – {percent:.2f}% confident"


# -------------------------
# Gradio Interface
# -------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=7, placeholder="Paste News Article Here"),
    outputs="text",
    title="📰 Dual-Branch Fake News Detector (BERT + Knowledge)",
    description="Enter a news article to check if it's real or fake using a BERT + TransE hybrid model."
)

if __name__ == "__main__":
    iface.launch()
