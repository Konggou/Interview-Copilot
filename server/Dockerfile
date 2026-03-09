FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
  # Force the CPU wheel so production images do not pull multi-GB CUDA bundles.
  && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu \
  && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

COPY server /app

RUN mkdir -p /app/data /app/temp/uploaded_files \
  && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
