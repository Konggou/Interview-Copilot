import hashlib
import os

from typing import List
from fastapi import UploadFile

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

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from utils.logger import logger


_EMBEDDINGS_CACHE = {}
_VECTORSTORE_CACHE = {}


def vectorstore_exists(persist_path: str) -> bool:
  exists = os.path.exists(persist_path) and bool(os.listdir(persist_path))
  logger.debug(f"Vectorstore exists at {persist_path}: {exists}")
  return exists


def get_embeddings(model_provider: str):
  model_provider = model_provider.lower()
  if model_provider in _EMBEDDINGS_CACHE:
    logger.debug(f"Using cached embeddings for provider: {model_provider}")
    return _EMBEDDINGS_CACHE[model_provider]

  if model_provider != "deepseek":
    logger.error(f"Unsupported LLM Provider: {model_provider}")
    raise ValueError(f"Unsupported LLM Provider: {model_provider}")

  logger.debug(f"Getting embeddings for provider: {model_provider}")
  embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
  )
  _EMBEDDINGS_CACHE[model_provider] = embedding
  return embedding


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
    logger.debug(f"VectorStore not found for provider: {model_provider}")
    raise ValueError(f"VectorStore not found for provider: {model_provider}")

  logger.debug(f"Loading existing vectorstore for provider: {model_provider}")
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


def retrieve_scored_chunks(
  model_provider: str,
  query: str,
  k: int = 3,
  threshold: float = SIMILARITY_THRESHOLD,
):
  results = find_similar_chunks(model_provider, query, k=k)
  top_score = results[0][1] if results else None
  return {
    "results": results,
    "top_score": top_score,
    # Chroma returns a distance-like score here: lower means more similar.
    "passes_threshold": bool(results) and top_score is not None and top_score <= threshold,
  }


def initialize_empty_vectorstores():
  logger.info("Initializing empty vectorstores...")

  if not DEEPSEEK_API_KEY:
    logger.debug("Skipping deepseek vectorstore init because DEEPSEEK_API_KEY is not set.")
    return

  provider = "deepseek"
  persist_path = VECTORSTORE_DIRECTORY[provider]
  os.makedirs(persist_path, exist_ok=True)

  if not os.listdir(persist_path):
    vectorstore = Chroma(
      embedding_function=get_embeddings(provider),
      persist_directory=persist_path,
    )
    _VECTORSTORE_CACHE[provider] = vectorstore
    logger.debug(f"Initialized vectorstore for {provider} at {persist_path}")

  logger.info("Vectorstore initialization complete.")


async def upsert_vectorstore_from_pdfs(uploaded_files: List[UploadFile], model_provider: str):
  model_provider = model_provider.lower()
  logger.debug(f"Upserting vectorstore for {model_provider}")

  file_paths = await save_uploaded_file(uploaded_files)
  docs = load_documents_from_paths(file_paths)
  chunks = split_documents_to_chunks(docs)
  chunks, chunk_ids = _prepare_chunks_for_upsert(chunks)

  persist_path = VECTORSTORE_DIRECTORY[model_provider]
  embedding = get_embeddings(model_provider)

  if vectorstore_exists(persist_path):
    logger.debug("Appending to existing vectorstore...")
    vectorstore = Chroma(
      persist_directory=persist_path,
      embedding_function=embedding,
    )
    vectorstore.add_documents(chunks, ids=chunk_ids)
    logger.debug(f"Added {len(chunks)} chunks to existing vectorstore.")
  else:
    vectorstore = Chroma.from_documents(
      documents=chunks,
      embedding=embedding,
      persist_directory=persist_path,
      ids=chunk_ids,
    )
    logger.debug(f"Created new vectorstore with {len(chunks)} chunks.")

  _VECTORSTORE_CACHE[model_provider] = vectorstore
  return vectorstore


def load_vectorstore(model_provider: str):
  model_provider = model_provider.lower()
  if model_provider in _VECTORSTORE_CACHE:
    logger.debug(f"Using cached vectorstore for provider: {model_provider}")
    return _VECTORSTORE_CACHE[model_provider]

  vectorstore = _create_vectorstore_instance(model_provider)
  _VECTORSTORE_CACHE[model_provider] = vectorstore
  return vectorstore


def get_collections_count(model_provider: str):
  logger.debug(f"Getting collection count for provider: {model_provider}")
  vectorstore = load_vectorstore(model_provider)
  return vectorstore._collection.count()


def find_similar_chunks(model_provider: str, query: str, k: int = 3):
  logger.debug(f"Searching for similar provider: {model_provider}")
  vectorstore = load_vectorstore(model_provider)
  results = vectorstore.similarity_search_with_score(query, k=k)
  for result, score in results:
    logger.debug(f"Similarity search result: {result}, score: {score}")
  return results
