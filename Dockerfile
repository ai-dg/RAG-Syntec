# slim base: faster to pull and to rebuild
FROM python:3.11-slim

WORKDIR /app

# Copy dependency manifest before the source code so Docker's layer cache
# only reinstalls dependencies when they actually change, not on every
# application code edit.
COPY pyproject.toml ./
RUN pip install uv --no-cache-dir && \
    uv pip install --system --no-cache -r pyproject.toml

COPY app ./app
COPY docs ./docs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
