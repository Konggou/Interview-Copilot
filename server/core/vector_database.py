import asyncio
import hashlib
import os

from fastapi import UploadFile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import (
  DEEPSEEK_API_KEY,
  SIMILARITY_THRESHOLD,
  SOURCE_SNIPPET_LENGTH,
  VECTORSTORE_DIRECTORY,
)
from core.document_processor import (
  load_documents_from_paths,
  save_uploaded_file,
  split_documents_to_chunks,
)
from utils.logger import logger


_EMBEDDINGS_CACHE = {}
_VECTORSTORE_CACHE = {}


def vectorstore_exists(persist_path: str) -> bool:
  exists = os.path.exists(persist_path) and bool(os.listdir(persist_path))
  logger.debug("vectorstore_exists", persist_path=persist_path, exists=exists)
  return exists


def get_embeddings(model_provider: str):
  model_provider = model_provider.lower()
  if model_provider in _EMBEDDINGS_CACHE:
    return _EMBEDDINGS_CACHE[model_provider]

  if model_provider != "deepseek":
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")

  embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
  )
  _EMBEDDINGS_CACHE[model_provider] = embedding
  return embedding


async def embed_query_text(model_provider: str, text: str) -> list[float]:
  embedding = get_embeddings(model_provider)
  return await asyncio.to_thread(embedding.embed_query, text)


def _normalize_source(raw_source: str) -> str:
  return os.path.basename(raw_source or "Unknown")


def _generate_chunk_id(doc, chunk_position: int) -> str:
  source = doc.metadata.get("source", "")
  page = doc.metadata.get("page", 0)
  chunk_index = doc.metadata.get("chunk_index", chunk_position)
  content_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
  return f"{_normalize_source(source)}:{page}:{chunk_index}:{content_hash}"


def _prepare_chunks_for_upsert(chunks):
  chunk_ids = []
  for index, chunk in enumerate(chunks):
    chunk_id = _generate_chunk_id(chunk, index)
    chunk.metadata["chunk_id"] = chunk_id
    chunk_ids.append(chunk_id)
  return chunks, chunk_ids


def _create_vectorstore_instance(model_provider: str):
  persist_path = VECTORSTORE_DIRECTORY[model_provider]
  if not vectorstore_exists(persist_path):
    raise ValueError(f"VectorStore not found for provider: {model_provider}")

  return Chroma(
    persist_directory=persist_path,
    embedding_function=get_embeddings(model_provider),
  )


def _trim_snippet(content: str, max_length: int) -> str:
  clean_content = " ".join((content or "").split())
  if len(clean_content) <= max_length:
    return clean_content
  return f"{clean_content[: max_length - 3].rstrip()}..."


def serialize_search_results(
  results_with_scores,
  snippet_length: int = SOURCE_SNIPPET_LENGTH,
):
  serialized_results = []
  for doc, score in results_with_scores:
    serialized_results.append({
      "source": _normalize_source(doc.metadata.get("source", "Unknown")),
      "page": doc.metadata.get("page", 0),
      "score": score,
      "snippet": _trim_snippet(doc.page_content, snippet_length),
      "metadata": doc.metadata,
      "page_content": doc.page_content,
    })
  return serialized_results


async def retrieve_scored_chunks(
  model_provider: str,
  query: str,
  k: int = 3,
  threshold: float = SIMILARITY_THRESHOLD,
):
  results = await find_similar_chunks(model_provider, query, k=k)
  top_score = results[0][1] if results else None
  return {
    "results": results,
    "top_score": top_score,
    # Chroma returns a distance-like score here: lower means more similar.
    "passes_threshold": bool(results) and top_score is not None and top_score <= threshold,
  }


async def initialize_empty_vectorstores():
  logger.info("Initializing empty vectorstores")

  if not DEEPSEEK_API_KEY:
    logger.warning("Skipping vectorstore initialization because DEEPSEEK_API_KEY is not set")
    return

  provider = "deepseek"
  persist_path = VECTORSTORE_DIRECTORY[provider]
  os.makedirs(persist_path, exist_ok=True)

  if not os.listdir(persist_path):
    vectorstore = await asyncio.to_thread(
      Chroma,
      embedding_function=get_embeddings(provider),
      persist_directory=persist_path,
    )
    _VECTORSTORE_CACHE[provider] = vectorstore

  logger.info("Vectorstore initialization complete")


async def upsert_vectorstore_from_pdfs(
  uploaded_files: list[UploadFile],
  model_provider: str,
):
  model_provider = model_provider.lower()
  file_paths = await save_uploaded_file(uploaded_files)
  docs = await asyncio.to_thread(load_documents_from_paths, file_paths)
  chunks = await asyncio.to_thread(split_documents_to_chunks, docs)
  chunks, chunk_ids = _prepare_chunks_for_upsert(chunks)

  persist_path = VECTORSTORE_DIRECTORY[model_provider]
  embedding = get_embeddings(model_provider)

  if vectorstore_exists(persist_path):
    vectorstore = await asyncio.to_thread(
      Chroma,
      persist_directory=persist_path,
      embedding_function=embedding,
    )
    await asyncio.to_thread(vectorstore.add_documents, chunks, ids=chunk_ids)
  else:
    vectorstore = await asyncio.to_thread(
      Chroma.from_documents,
      documents=chunks,
      embedding=embedding,
      persist_directory=persist_path,
      ids=chunk_ids,
    )

  _VECTORSTORE_CACHE[model_provider] = vectorstore
  return vectorstore


async def load_vectorstore(model_provider: str):
  model_provider = model_provider.lower()
  if model_provider in _VECTORSTORE_CACHE:
    return _VECTORSTORE_CACHE[model_provider]

  vectorstore = await asyncio.to_thread(_create_vectorstore_instance, model_provider)
  _VECTORSTORE_CACHE[model_provider] = vectorstore
  return vectorstore


async def get_collections_count(model_provider: str):
  vectorstore = await load_vectorstore(model_provider)
  return await asyncio.to_thread(vectorstore._collection.count)


async def find_similar_chunks(model_provider: str, query: str, k: int = 3):
  vectorstore = await load_vectorstore(model_provider)
  return await asyncio.to_thread(vectorstore.similarity_search_with_score, query, k)
