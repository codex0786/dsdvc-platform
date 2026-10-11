FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt
COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY legacy ./legacy
COPY tests ./tests
COPY scripts ./scripts
RUN pip install -e . && useradd -m dsdvc && chown -R dsdvc /app
USER dsdvc
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
