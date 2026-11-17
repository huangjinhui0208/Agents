"""Provider-independent vector embeddings for repository notes."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from openai import OpenAI


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    """Create normalized vectors locally or through an OpenAI-compatible API.

    ``local_hash`` is a dependency-free feature-hashing encoder. It provides real
    vector/cosine retrieval immediately, while ``openai_compatible`` lets the user
    switch to a semantic embedding model without changing NoteStore.
    """

    def __init__(
        self,
        *,
        provider: str = "local_hash",
        model: str = "feature-hash-v1",
        base_url: str = "",
        api_key: str = "",
        dimensions: int = 384,
        timeout_s: float = 60.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self._client: OpenAI | None = None
        if provider == "openai_compatible":
            self._client = OpenAI(
                api_key=api_key or "local-placeholder",
                base_url=base_url,
                timeout=timeout_s,
            )
        elif provider != "local_hash":
            raise ValueError(f"不支持的向量提供方：{provider}")

    @property
    def signature(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    @staticmethod
    def _features(text: str) -> Iterable[str]:
        lowered = text.casefold()
        for token in re.findall(r"[a-z_][a-z0-9_./:-]*|[\u3400-\u9fff]+", lowered):
            yield f"token:{token}"
            if re.fullmatch(r"[\u3400-\u9fff]+", token):
                for size in (2, 3):
                    for index in range(max(0, len(token) - size + 1)):
                        yield f"cjk{size}:{token[index:index + size]}"
            else:
                parts = [part for part in re.split(r"[_./:-]+", token) if part]
                for part in parts:
                    yield f"part:{part}"
                    if len(part) >= 4:
                        for index in range(len(part) - 2):
                            yield f"tri:{part[index:index + 3]}"

    def _local_vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if raw & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "local_hash":
            return [self._local_vector(text) for text in texts]
        assert self._client is not None
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise EmbeddingError(
                f"向量模型调用失败：provider={self.provider}, model={self.model}"
            ) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [self._normalize(list(item.embedding)) for item in ordered]
        if len(vectors) != len(texts):
            raise EmbeddingError("向量模型返回数量与输入数量不一致。")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError("向量模型返回了不一致的维度。")
        return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    return sum(a * b for a, b in zip(left, right))


__all__ = ["EmbeddingClient", "EmbeddingError", "cosine_similarity"]
