
"""Generate frozen DistilBERT embeddings for maintenance logs."""

from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EMBEDDING_DIM = 768


class TextEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
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
