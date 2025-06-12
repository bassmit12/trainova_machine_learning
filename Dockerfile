FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create volume mount points for data persistence
VOLUME ["/app/trainova_ml/data/models", "/app/trainova_ml/data/datasets"]

# Command to run the application
ENTRYPOINT ["python", "trainova-cli.py"]