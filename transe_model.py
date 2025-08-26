# transe_model.py
# Simple TransE model + an Aggregator used at inference time.
# Put this file in the SAME folder as gui_fusion_demo.py

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """
    Minimal TransE for inference.

    Parameters
    ----------
    num_entities : int
        Size of the entity vocabulary.
    num_relations : int
        Size of the relation vocabulary.
    dim : int, optional
        Embedding dimension (default: 100). Kept for compatibility.
    embedding_dim : int, optional
        Alternative keyword for embedding dimension. If provided, overrides `dim`.

    Notes
    -----
    Scoring function: -||h + r - t||_2
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 100,
        embedding_dim: Optional[int] = None
    ):
        super().__init__()
        # Allow both dim and embedding_dim
        if embedding_dim is not None:
            dim = embedding_dim

        self.dim = dim
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Xavier init then L2-normalize (common in TransE)
        nn.init.xavier_uniform_(self.entity_emb.weight.data)
        nn.init.xavier_uniform_(self.relation_emb.weight.data)
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)
            self.relation_emb.weight.data = F.normalize(self.relation_emb.weight.data, p=2, dim=1)

    def score_triple(
        self,
        h_idx: torch.LongTensor,
        r_idx: torch.LongTensor,
        t_idx: torch.LongTensor
    ) -> torch.Tensor:
        """
        Score a batch of triples.

        Returns
        -------
        torch.Tensor
            Shape (N,), higher is better.
        """
        h = self.entity_emb(h_idx)      # (N, d)
        r = self.relation_emb(r_idx)    # (N, d)
        t = self.entity_emb(t_idx)      # (N, d)
        return -(h + r - t).pow(2).sum(dim=1).sqrt()

    def entity_vector(self, idx: int) -> torch.Tensor:
        """Return a single entity embedding vector (1, dim), L2-normalized."""
        vec = self.entity_emb.weight[idx].unsqueeze(0)
        return F.normalize(vec, p=2, dim=1)

    def relation_vector(self, idx: int) -> torch.Tensor:
        """Return a single relation embedding vector (1, dim), L2-normalized."""
        vec = self.relation_emb.weight[idx].unsqueeze(0)
        return F.normalize(vec, p=2, dim=1)


class Aggregator(nn.Module):
    """
    Turns extracted triplets into ONE knowledge vector for an article.

    Strategy (simple & robust):
      - Map (head, tail) entity strings via `entity_vocab` to ids.
      - Grab their embeddings from TransE.
      - Mean-pool the entity vectors.
      - Optional linear projection to `out_dim` (defaults to no change).

    This is intentionally minimal to keep GUI inference snappy and reliable.
    """
    def __init__(self, dim: int = 100, out_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.proj = nn.Linear(dim, self.out_dim) if self.out_dim != self.dim else nn.Identity()

    def forward(
        self,
        triplets: List[Tuple[str, str, str]],
        entity_vocab: Dict[str, int],
        relation_vocab: Dict[str, int],  # kept for API symmetry; not required by this strategy
        transe: TransE,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        triplets : list of (head, relation, tail)
        entity_vocab : dict str->int
        relation_vocab : dict str->int (unused in mean-pool strategy, but kept for compatibility)
        transe : TransE
        device : optional torch.device

        Returns
        -------
        torch.Tensor
            (1, out_dim) pooled knowledge vector. Zeros if no entities were mapped.
        """
        device = device or next(transe.parameters()).device
        entity_vecs = []

        for h, r, t in triplets:
            h_id = entity_vocab.get(h)
            t_id = entity_vocab.get(t)

            if h_id is not None:
                entity_vecs.append(transe.entity_vector(h_id).to(device))
            if t_id is not None:
                entity_vecs.append(transe.entity_vector(t_id).to(device))

        if len(entity_vecs) == 0:
            # No mapped entities -> return zeros (so fusion still runs)
            return torch.zeros(1, self.out_dim, device=device)

        M = torch.cat(entity_vecs, dim=0)        # (K, dim)
        pooled = M.mean(dim=0, keepdim=True)     # (1, dim)
        pooled = F.normalize(pooled, p=2, dim=1) # keep scale stable
        return self.proj(pooled)