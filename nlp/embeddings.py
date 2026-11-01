
"""Generate frozen DistilBERT embeddings for maintenance logs.

The transformer dependency is intentionally lazy. Rule-based NLP remains
usable when the optional model stack is unavailable, while embedding calls
fail with a precise offline diagnostic instead of an import-time traceback.
"""

from pathlib import Path
from typing import Any, List

import torch

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EMBEDDING_DIM = 768


class TextEmbedder:
    def __init__(self):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLP embeddings unavailable: optional 'transformers' is not installed. "
                "Install a pinned environment and provide the approved model/tokenizer "
                "artifacts; no download is attempted by this project."
            ) from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            self.model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - normalize cache/model failures.
            raise RuntimeError(
                f"NLP embeddings unavailable: local {MODEL_NAME!r} artifacts are not complete. "
                "No network download is attempted; provide pinned model artifacts explicitly."
            ) from exc
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)

        safe_texts = [
            text if isinstance(text, str) else ""
            for text in texts
        ]

        inputs = self.tokenizer(
            safe_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state

        # Mean pooling excludes padding tokens.
        mask = inputs["attention_mask"].unsqueeze(-1).to(
            token_embeddings.dtype
        )
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts

        return embeddings.cpu().float()


def generate_embeddings(texts: List[str]) -> torch.Tensor:
    embedder = TextEmbedder()
    return embedder.encode(texts)


def embedding_diagnostics(cache_root: str | Path | None = None) -> dict[str, Any]:
    """Report optional embedding readiness without importing or downloading models."""
    try:
        import importlib.util
        transformers_available = importlib.util.find_spec("transformers") is not None
    except Exception:  # pragma: no cover - defensive environment probing.
        transformers_available = False
    root = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    model_markers = list(root.glob("**/*distilbert*")) if root.exists() else []
    return {
        "transformers_available": transformers_available,
        "model_name": MODEL_NAME,
        "local_cache_root": str(root),
        "local_model_markers_found": bool(model_markers),
        "ready": bool(transformers_available and model_markers),
        "network_download_attempted": False,
        "classification": "optional NLP embedding dependency; not required for rule-based fixture tests",
    }
