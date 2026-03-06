import os
import aiofiles

from typing import List
from fastapi import UploadFile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

from config.settings import TEMPFILE_UPLOAD_DIRECTORY
from utils.logger import logger


def validate_pdf(file: UploadFile, max_size_mb: int = 200):
  """
  功能: 校验上传的 PDF 文件格式和大小
  参数:
    - file: FastAPI UploadFile 文件对象
    - max_size_mb: 允许的最大文件大小 (单位 MB，默认 200)
  返回值: 无 (如果不合法则抛出 ValueError)
  """
  if not file.filename.endswith(".pdf"):
    logger.warning(f"Invalid file type: {file.filename}")
    raise ValueError(f"{file.filename} is not a valid PDF file.")

  file_size_mb = len(file.file.read()) / (1024 * 1024)
  file.file.seek(0)

  if file_size_mb > max_size_mb:
    logger.warning(f"File too large: {file.filename} ({file_size_mb:.2f} MB)")
    raise ValueError(f"PDF file size exceeds the maximum allowed size of {max_size_mb} MB.")

  logger.debug(f"Validated PDF: {file.filename} ({file_size_mb:.2f} MB)")

async def save_uploaded_file(files: List[UploadFile]) -> List[str]:
  """
  功能: 异步将上传的文件保存到服务器临时目录
  参数:
    - files: UploadFile 文件对象列表
  返回值:
    - List[str]: 保存后的本地文件绝对路径列表
  """
  os.makedirs(TEMPFILE_UPLOAD_DIRECTORY, exist_ok=True)
  file_paths = []

  for file in files:
    validate_pdf(file)
    file_path = os.path.join(TEMPFILE_UPLOAD_DIRECTORY, file.filename)
    async with aiofiles.open(file_path, "wb") as f:
      content = await file.read()
      await f.write(content)
    file_paths.append(file_path)
    logger.debug(f"Saved uploaded file: {file.filename} to {file_path}")

  return file_paths

def load_documents_from_paths(file_paths: List[str]):
  """
  功能: 使用 loader 加载本地 PDF 文件并转换为 Document 对象
  参数:
    - file_paths: PDF 文件的路径列表
  返回值:
    - List[Document]: 加载后的 LangChain Document 对象列表
  """
  docs = []
  for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    loaded = loader.load()
    normalized_source = os.path.basename(file_path)
    for page_index, doc in enumerate(loaded):
      doc.metadata["source"] = normalized_source
      doc.metadata["page"] = doc.metadata.get("page", page_index)
    logger.debug(f"Loaded {len(loaded)} documents from {file_path}")
    docs.extend(loaded)

  return docs

def split_documents_to_chunks(docs) -> List[str]:
  """
  功能: 将文档长文本拆分为较小的文本块 (Chunks)
  参数:
    - docs: Document 对象列表
  返回值:
    - List[Document]: 拆分后的 Document Chunks 列表
  """
  text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
  chunks = text_splitter.split_documents(docs)
  page_chunk_counts = {}

  for chunk in chunks:
    source = chunk.metadata.get("source", "Unknown")
    page = chunk.metadata.get("page", 0)
    page_key = f"{source}:{page}"
    chunk_index = page_chunk_counts.get(page_key, 0)
    page_chunk_counts[page_key] = chunk_index + 1
    chunk.metadata["source"] = os.path.basename(source)
    chunk.metadata["page"] = page
    chunk.metadata["chunk_index"] = chunk_index

  logger.debug(f"Split {len(docs)} docs into {len(chunks)} chunks")
  return chunks
