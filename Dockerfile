# Use Python 3.12 to match pyproject requirement
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install uv (the tool you use to sync dependencies)
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Some build backends require the README referenced in pyproject.toml to exist
# (hatchling validates the readme during editable builds). Copy it before
# running `uv sync --frozen` so the build step can find the file.
COPY README.md ./

# Install dependencies from uv.lock (frozen ensures pinned)
RUN uv sync --frozen

# Copy application code (only the package and any data)
COPY poc ./poc
COPY poc/data ./data

# If you have other scripts or entry points, copy those too
COPY main.py ./

# Run main.py as the entry point
CMD ["python", "main.py"]