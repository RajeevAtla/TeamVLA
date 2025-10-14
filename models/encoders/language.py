"""Language encoder builders and utilities for TeamVLA."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def build_text_encoder(
    name: str = "gru",
    *,
    vocab_size: int = 256,
    d_model: int = 128,
    n_layers: int = 1,
) -> Any:
    """Build a lightweight text encoder."""

    if name != "gru":  # pragma: no cover
        raise ValueError(f"Unsupported text encoder '{name}'.")
    _require_torch()
    return _GRUTextEncoder(vocab_size=vocab_size, hidden_size=d_model, num_layers=n_layers)


def tokenize(texts: Iterable[str], vocab: Mapping[str, int] | None = None, max_length: int = 32) -> dict[str, Any]:
    """Tokenize raw text into integer IDs and attention masks."""

    _require_torch()
    vocab = vocab or _default_vocab()
    tokens = [_encode_text(text, vocab, max_length) for text in texts]
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

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        _require_torch()
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, hidden_size)
        self._gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeddings = self._embedding(input_ids)
        packed, _ = self._gru(embeddings)
        masked = packed * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return pooled


def _encode_text(text: str, vocab: Mapping[str, int], max_length: int) -> dict[str, torch.Tensor]:
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in text.lower()[:max_length]]
    ids += [vocab["<pad>"]] * (max_length - len(ids))
    input_ids = torch.tensor(ids, dtype=torch.long)
    attention = (input_ids != vocab["<pad>"]).to(torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention}


def _default_vocab() -> Mapping[str, int]:
    letters = {chr(c): i + 2 for i, c in enumerate(range(ord("a"), ord("z") + 1))}
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update(letters)
    vocab[" "] = len(vocab)
    return vocab
