"""Pinecone helper wrapper for index creation and record upsert."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
from pinecone import Pinecone
from ..config import Config


class PineconeWrapper:
	"""Thin Pinecone wrapper used by the RAG pipeline."""

	def __init__(self, api_key: Optional[str] = None, default_namespace: str = "mbse-demo"):
		resolved_api_key = api_key or Config.PINECONE_API_KEY
		if not resolved_api_key:
			raise ValueError("Pinecone API key is required. Set Config.PINECONE_API_KEY or pass api_key.")

		self._pc = Pinecone(api_key=resolved_api_key)
		self.default_namespace = default_namespace

	def create_index(
		self,
		index_name: str,
		cloud: str = "aws",
		region: str = "us-east-1",
		embedding_model: str = "llama-text-embed-v2",
		text_field: str = "chunk_text",
	) -> bool:
		"""Create an integrated-embedding index if it does not exist."""
		if self._pc.has_index(index_name):
			return False

		self._pc.create_index_for_model(
			name=index_name,
			cloud=cloud,
			region=region,
			embed={
				"model": embedding_model,
				"field_map": {"text": text_field},
			},
		)
		return True

	def upsert(
		self,
		index_name: str,
		records: Sequence[Dict[str, Any]],
		namespace: Optional[str] = None,
	) -> int:
		"""Upsert records into an integrated-embedding Pinecone index."""
		if not records:
			return 0

		index = self._pc.Index(index_name)
		target_namespace = namespace or self.default_namespace

		index.upsert_records(
			namespace=target_namespace,
			records=list(records),
		)
		return len(records)

	def search(
		self,
		index_name: str,
		query_text: str,
		top_k: int = 3,
		namespace: Optional[str] = None,
		filter_dict: Optional[Dict[str, Any]] = None,
		fields: Optional[List[str]] = None,
	) -> Dict[str, Any]:
		"""Search the index with integrated embedding and optional metadata filtering."""
		index = self._pc.Index(index_name)
		target_namespace = namespace or self.default_namespace

		results = index.search(
			namespace=target_namespace,
			top_k=top_k,
			inputs={"text": query_text},
			filter=filter_dict or None,
			fields=fields,
		)
		return results

	@property
	def client(self) -> Pinecone:
		"""Expose underlying Pinecone client when low-level access is needed."""
		return self._pc
