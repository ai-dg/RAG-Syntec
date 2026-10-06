# slim base: faster to pull and to rebuild
FROM python:3.11-slim AS base

WORKDIR /app

# Copy dependency manifest before the source code so Docker's layer cache
# only reinstalls dependencies when they actually change, not on every
# application code edit.
COPY pyproject.toml ./
RUN pip install uv --no-cache-dir && \
    uv pip install --system --no-cache -r pyproject.toml

FROM base AS app

COPY app ./app
COPY data ./data
COPY models ./models

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS demo

RUN uv pip install --system --no-cache "gradio>=5.0"
COPY demo ./demo

EXPOSE 7860

CMD ["python", "demo/app.py"]
