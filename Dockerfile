FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=5010

# Create volume mount points for data persistence
VOLUME ["/app/trainova_ml/data/models", "/app/trainova_ml/data/datasets"]

# Expose the port the app runs on
EXPOSE 5010

# Command to run the production server
CMD ["python", "run_production.py"]