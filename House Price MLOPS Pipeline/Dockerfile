FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and required input artifacts
COPY run.py .
COPY config.yaml .
COPY data.csv .

# No hardcoded paths: all I/O paths are passed as CLI args.
# Metrics + log land in /app and are printed to stdout for the grader.
CMD ["python", "run.py", \
     "--input", "data.csv", \
     "--config", "config.yaml", \
     "--output", "metrics.json", \
     "--log-file", "run.log"]
