
"""Generate frozen DistilBERT embeddings for maintenance logs.

The transformer dependency is intentionally lazy. Rule-based NLP remains
usable when the optional model stack is unavailable, while embedding calls
fail with a precise offline diagnostic instead of an import-time traceback.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import TYPE_CHECKING, Any, List, Sequence

if TYPE_CHECKING:
    import torch

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EMBEDDING_DIM = 768
DEFAULT_CACHE_DIR = Path(os.environ.get("IOT_NLP_CACHE_DIR", Path.home() / ".cache" / "huggingface"))


class TextEmbedder:
    def __init__(self, model_path: str | Path | None = None, *, device: str | None = None,
                 max_length: int = MAX_LENGTH, batch_size: int = 16):
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.model_path = str(model_path or MODEL_NAME)
        self.max_length = max_length
        self.batch_size = batch_size
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLP embeddings unavailable: optional 'transformers' is not installed. "
                "Install a pinned environment and provide the approved model/tokenizer "
                "artifacts; no download is attempted by this project."
            ) from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=str(DEFAULT_CACHE_DIR), local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_path, cache_dir=str(DEFAULT_CACHE_DIR), local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - normalize cache/model failures.
            raise RuntimeError(
                f"NLP embeddings unavailable: local {self.model_path!r} artifacts are not complete. "
                "No network download is attempted; provide pinned model artifacts explicitly."
            ) from exc
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "NLP embeddings unavailable: optional 'torch' is not installed. "
                "No network download is attempted."
            ) from exc
        self.model.eval()
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        try:
            self.device = torch.device(requested)
            self.model.to(self.device)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Unable to use embedding device {requested!r}.") from exc

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        import torch

        if not texts:
            return torch.empty((0, EMBEDDING_DIM), dtype=torch.float32)
        if not isinstance(texts, (list, tuple)) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a sequence of strings")
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty or whitespace-only values")
        with torch.inference_mode():
            batches = []
            for start in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer(list(texts[start:start + self.batch_size]), padding=True,
                                        truncation=True, max_length=self.max_length, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(pooled.cpu().float())
            result = torch.cat(batches, dim=0)
        if result.shape != (len(texts), EMBEDDING_DIM) or not torch.isfinite(result).all().item():
            raise RuntimeError("Transformer returned an invalid embedding tensor")
        return result


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
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_DIR
    model_markers = list(root.glob("**/*distilbert*")) if root.exists() else []
    local_model_ready = False
    diagnostic_error = None
    if transformers_available and model_markers:
        try:
            from transformers import AutoModel, AutoTokenizer
            AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(root), local_files_only=True)
            AutoModel.from_pretrained(MODEL_NAME, cache_dir=str(root), local_files_only=True)
            local_model_ready = True
        except Exception as exc:  # noqa: BLE001 - diagnostics must not download.
            diagnostic_error = str(exc)
    return {
        "transformers_available": transformers_available,
        "model_name": MODEL_NAME,
        "local_cache_root": str(root),
        "local_model_markers_found": bool(model_markers),
        "ready": local_model_ready,
        "diagnostic_error": diagnostic_error,
        "network_download_attempted": False,
        "classification": "optional NLP embedding dependency; not required for rule-based fixture tests",
    }
