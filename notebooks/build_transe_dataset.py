import json
import os
import torch

# Paths to your triplet JSON files
paths = {
    "train": "data/triplets/triplets_train.json",
    "val": "data/triplets/triplets_val.json",
    "test": "data/triplets/triplets_test.json"
}

# Vocab class
class Vocab:
    def __init__(self):
        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = {}
        self.id2rel = {}
        self.ent_counter = 0
        self.rel_counter = 0

    def add(self, h, r, t):
        for ent in [h, t]:
            if ent not in self.ent2id:
                self.ent2id[ent] = self.ent_counter
                self.id2ent[self.ent_counter] = ent
                self.ent_counter += 1
        if r not in self.rel2id:
            self.rel2id[r] = self.rel_counter
            self.id2rel[self.rel_counter] = r
            self.rel_counter += 1

    def get_ids(self, h, r, t):
        return self.ent2id[h], self.rel2id[r], self.ent2id[t]

# Normalize triplet to (head, relation, tail)
def normalize_triplet(triplet):
    # Heuristic: If middle element looks like a relation, assume correct
    # Otherwise, swap position
    h, r, t = triplet
    if r.lower() in {"position held", "member of", "facet of", "candidate", "point in time", "part of", "field of work",
                     "located in", "applies to jurisdiction", "officeholder", "has part", "participant",
                     "country", "uses", "location", "notable work", "discoverer or inventor"}:
        return h, r, t
    elif t.lower() in {"position held", "member of", "facet of", "candidate", "point in time", "part of", "field of work"}:
        return h, t, r
    else:
        return h, r, t  # Fallback

# Load and process triplets
def process_triplets(paths):
    vocab = Vocab()
    triplet_data = {}

    for split, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        id_triplets = []
        for item in raw_data:
            for triplet in item["triplets"]:
                norm_triplet = normalize_triplet(triplet)

                # Filter out noisy data
                if "extract triples" in norm_triplet or "" in norm_triplet:
                    continue

                h, r, t = norm_triplet
                vocab.add(h, r, t)
                id_triplets.append(vocab.get_ids(h, r, t))

        triplet_data[split] = id_triplets
        print(f"{split.title()} samples: {len(id_triplets)}")

    return triplet_data, vocab

# Process
triplet_data, vocab = process_triplets(paths)

# Save vocab
os.makedirs("models", exist_ok=True)
torch.save(vocab.ent2id, "models/entity_vocab.pt")
torch.save(vocab.rel2id, "models/relation_vocab.pt")

# Save datasets
torch.save(triplet_data["train"], "models/transe_train.pt")
torch.save(triplet_data["val"], "models/transe_val.pt")
torch.save(triplet_data["test"], "models/transe_test.pt")

# Print stats
print(f"Total Entities: {len(vocab.ent2id)}")
print(f"Total Relations: {len(vocab.rel2id)}")
