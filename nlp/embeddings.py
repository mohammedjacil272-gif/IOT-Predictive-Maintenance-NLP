"""Offline-first, frozen DistilBERT embeddings for maintenance text."""

from functools import lru_cache
import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
from nlp.preprocessing import MAX_TEXT_CHARS

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EMBEDDING_DIM = 768
POOLING = "attention-mask mean pooling"


class ModelUnavailableError(RuntimeError):
    """Raised when approved local transformer weights cannot be loaded."""


def _model_options() -> dict:
    options = {"local_files_only": os.environ.get("IOT_NLP_ALLOW_DOWNLOAD") != "1"}
    cache_dir = os.environ.get("IOT_NLP_CACHE_DIR")
    if cache_dir:
        options["cache_dir"] = str(Path(cache_dir).expanduser())
    return options


@lru_cache(maxsize=2)
def _load_model(model_name: str, cache_dir: str | None, local_files_only: bool):
    source = model_name
    options = {"local_files_only": local_files_only}
    if cache_dir and (Path(cache_dir) / "config.json").is_file():
        source = cache_dir
    elif cache_dir:
        options["cache_dir"] = cache_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **options)
        model = AutoModel.from_pretrained(source, **options)
    except (OSError, ValueError) as exc:
        raise ModelUnavailableError(
            f"Local model {model_name!r} is unavailable. Set IOT_NLP_CACHE_DIR "
            "to a directory containing its tokenizer and weights. Downloads "
            "require explicit IOT_NLP_ALLOW_DOWNLOAD=1."
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


class TextEmbedder:
    """Encode text with cached, frozen DistilBERT weights."""

    def __init__(self, tokenizer=None, model=None):
        self.tokenizer = tokenizer
        self.model = model
        self.load_error = None
        if tokenizer is None or model is None:
            options = _model_options()
            cache_dir = options.get("cache_dir")
            try:
                self.tokenizer, self.model = _load_model(
                    MODEL_NAME,
                    str(cache_dir) if cache_dir else None,
                    options["local_files_only"],
                )
            except ModelUnavailableError as exc:
                self.load_error = exc

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple of strings")
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("every text value must be a string")
        if any(len(text) > MAX_TEXT_CHARS for text in texts):
            raise ValueError(f"text cannot exceed {MAX_TEXT_CHARS} characters")
        if self.load_error is not None or self.tokenizer is None or self.model is None:
            raise ModelUnavailableError(str(self.load_error or "Model is not loaded"))
        inputs = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        if "attention_mask" not in inputs:
            raise ValueError("Tokenizer output is missing attention_mask")
        if "input_ids" not in inputs:
            raise ValueError("Tokenizer output is missing input_ids")
        input_ids = inputs["input_ids"]
        mask_values = inputs["attention_mask"]
        if (input_ids.ndim != 2 or input_ids.shape[0] != len(texts)
                or mask_values.ndim != 2 or tuple(mask_values.shape) != tuple(input_ids.shape)):
            raise ValueError("Tokenizer returned an invalid attention_mask")
        if not bool(((mask_values == 0) | (mask_values == 1)).all()):
            raise ValueError("attention_mask must contain only zero and one")
        if not bool((mask_values.sum(dim=1) > 0).all()):
            raise ValueError("attention_mask must include at least one token per input")

        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        if (token_embeddings.ndim != 3 or token_embeddings.shape[0] != len(texts)
                or token_embeddings.shape[2] != EMBEDDING_DIM
                or token_embeddings.shape[1] != mask_values.shape[1]):
            raise ValueError("Transformer output has an unexpected shape")

        mask = mask_values.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.cpu().float()
        empty_rows = [index for index, text in enumerate(texts) if not text.strip()]
        if empty_rows:
            pooled[empty_rows] = 0
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("Transformer generated non-finite embeddings")
        return pooled


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    """Return one process-wide embedder; weights are loaded at most once."""
    return TextEmbedder()


def generate_embeddings(texts: List[str]) -> torch.Tensor:
    return get_text_embedder().encode(texts)
