"""Language encoder builders and utilities for TeamVLA."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class TextEncoderConfig:
    """Configuration for the default GRU-based text encoder."""

    vocab_size: int = 512
    d_model: int = 128
    num_layers: int = 1
    dropout: float = 0.0


def build_text_encoder(
    name: str = "gru",
    *,
    vocab_size: int = 512,
    d_model: int = 128,
    n_layers: int = 1,
    dropout: float = 0.0,
    config: TextEncoderConfig | None = None,
) -> Any:
    """Build a lightweight text encoder."""

    if name != "gru":  # pragma: no cover - other variants can be added later
        raise ValueError(f"Unsupported text encoder '{name}'.")
    cfg = config or TextEncoderConfig(vocab_size=vocab_size, d_model=d_model, num_layers=n_layers, dropout=dropout)
    cfg = TextEncoderConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    _require_torch()
    return _GRUTextEncoder(cfg)


def tokenize(
    texts: Iterable[str],
    vocab: Mapping[str, int] | None = None,
    max_length: int = 32,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> dict[str, Any]:
    """Tokenize raw text into integer IDs and attention masks."""

    _require_torch()
    vocab = vocab or _default_vocab(pad_token=pad_token, unk_token=unk_token)
    tokens = [_encode_text(text, vocab, max_length, pad_token=pad_token, unk_token=unk_token) for text in texts]
    input_ids = torch.stack([token["input_ids"] for token in tokens])
    attention = torch.stack([token["attention_mask"] for token in tokens])
    return {"input_ids": input_ids, "attention_mask": attention}


def forward_text(encoder: Any, tokens: Mapping[str, Any]) -> Any:
    """Forward tokens through the text encoder."""

    _require_torch()
    return encoder(tokens["input_ids"], tokens["attention_mask"])


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("PyTorch is required to use the language encoders.")


class _GRUTextEncoder(nn.Module if nn is not None else object):
    """Gated recurrent unit encoder producing pooled features."""

    def __init__(self, cfg: TextEncoderConfig) -> None:
        _require_torch()
        super().__init__()
        self._embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self._gru = nn.GRU(
            cfg.d_model,
            cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self._dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeddings = self._embedding(input_ids)
        packed, _ = self._gru(embeddings)
        masked = packed * attention_mask.unsqueeze(-1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = masked.sum(dim=1) / denom
        return self._dropout(pooled)


def _encode_text(
    text: str,
    vocab: Mapping[str, int],
    max_length: int,
    *,
    pad_token: str,
    unk_token: str,
) -> dict[str, torch.Tensor]:
    tokens: list[int] = [vocab.get(ch.lower(), vocab[unk_token]) for ch in text[:max_length]]
    pad_id = vocab[pad_token]
    if len(tokens) < max_length:
        tokens.extend([pad_id] * (max_length - len(tokens)))
    input_ids = torch.tensor(tokens, dtype=torch.long)
    attention_mask = (input_ids != pad_id).float()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _default_vocab(*, pad_token: str, unk_token: str) -> Mapping[str, int]:
    letters = {chr(c): i + 2 for i, c in enumerate(range(ord("a"), ord("z") + 1))}
    vocab: dict[str, int] = {pad_token: 0, unk_token: 1}
    vocab.update(letters)
    vocab[" "] = len(vocab)
    vocab["."] = len(vocab)
    vocab[","] = len(vocab)
    return vocab

