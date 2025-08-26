# triplet_extraction/rebel_triplet_extractor.py
# Minimal, self-contained REBEL wrapper with lazy loading + robust parsing.

from typing import List, Tuple
from functools import lru_cache

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except Exception as e:
    raise RuntimeError(
        "transformers/torch not available. Install with:\n"
        "  pip install --upgrade torch transformers sentencepiece accelerate"
    ) from e


MODEL_NAME = "Babelscape/rebel-large"  # T5-style model that outputs triplets in text


@lru_cache(maxsize=1)
def _load_rebel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tok, model, device


def _parse_rebel_output(text: str) -> List[Tuple[str, str, str]]:
    """
    REBEL typically emits sequences like:
      "<triplet> Barack Obama <subj> country <obj> United States <triplet> ..."
    But sometimes emits ' | ' separated triplets.
    We support both formats and deduplicate results.
    """
    triplets = []
    seen = set()

    # 1) Angle-bracket tagged format
    import re
    pattern = re.compile(
        r"(?:<triplet>\s*)(.+?)\s*<subj>\s*(.+?)\s*<obj>\s*(.+?)(?=(?:\s*<triplet>|\s*$))",
        flags=re.IGNORECASE | re.DOTALL
    )
    for h, r, t in pattern.findall(text):
        h, r, t = h.strip(), r.strip(), t.strip()
        key = (h, r, t)
        if all(key) and key not in seen:
            seen.add(key)
            triplets.append(key)

    # 2) Pipe-separated fallback: subject | relation | object
    if not triplets:
        chunks = [c.strip() for c in text.split("<triplet>") if c.strip()]
        for chunk in chunks:
            parts = [p.strip() for p in chunk.split("|")]
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], "|".join(parts[2:]).strip()
                key = (h, r, t)
                if all(key) and key not in seen:
                    seen.add(key)
                    triplets.append(key)

    return triplets


def extract_triplets(article_text: str, max_new_tokens: int = 128) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (head, relation, tail) triplets extracted from raw text.
    This is what your GUI expects to feed into the knowledge aggregator. :contentReference[oaicite:1]{index=1}
    """
    if not article_text or not article_text.strip():
        return []

    tok, model, device = _load_rebel()
    inputs = tok([article_text], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            num_beams=3,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )

    decoded = tok.batch_decode(gen, skip_special_tokens=True)
    # Collect triplets from all beams/hypotheses (usually 1 here)
    results = []
    for seq in decoded:
        results.extend(_parse_rebel_output(seq))

    return results