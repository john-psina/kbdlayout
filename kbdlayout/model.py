"""
Char-level TextCNN classifier for the keyboard-layout task.

4 output classes (intent_lang + actual_layout):
    uk_uk, uk_en, en_en, en_uk
"""

from collections import Counter
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

LABELS: list[str] = ["uk_uk", "uk_en", "en_en", "en_uk"]
LABEL_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(LABELS)}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CharVocab:
    def __init__(self, chars: list[str]):
        self.itos: list[str] = [PAD_TOKEN, UNK_TOKEN] + chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.pad_idx = 0
        self.unk_idx = 1

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(c, self.unk_idx) for c in text[:max_len]]
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "CharVocab":
        v = cls.__new__(cls)
        v.itos = d["itos"]
        v.stoi = {c: i for i, c in enumerate(v.itos)}
        v.pad_idx = 0
        v.unk_idx = 1
        return v

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 5) -> "CharVocab":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(t)
        chars = sorted(c for c, n in counter.items() if n >= min_freq)
        return cls(chars)


class LayoutClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 4,
        emb_dim: int = 32,
        n_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)  # (B, E, L)
        feats = []
        for conv in self.convs:
            # pad if input shorter than kernel
            if emb.size(2) < conv.kernel_size[0]:
                pad = conv.kernel_size[0] - emb.size(2)
                e = F.pad(emb, (0, pad))
            else:
                e = emb
            c = F.relu(conv(e))
            p = c.max(dim=2)[0]
            feats.append(p)
        cat = torch.cat(feats, dim=1)
        return self.fc(self.dropout(cat))
