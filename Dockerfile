# Stage 1: Build dependencies
FROM python:3.10-alpine as builder

# Environment variables to reduce clutter and avoid cache issues
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required for build
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev \
    cargo \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# Copy dependencies files
COPY pyproject.toml poetry.lock ./

# Install only runtime dependencies
RUN poetry lock && \
    poetry install --only main --no-interaction --no-ansi

# Stage 2: Final lightweight image
FROM python:3.10-alpine

WORKDIR /app

# Set PYTHONPATH so the app can find modules
ENV PYTHONPATH=/app

# Install runtime dependencies
RUN apk add --no-cache libffi openssl

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source code
COPY . .

# Expose the app port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]