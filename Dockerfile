# Use official Python lightweight image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY app.py .
COPY train.py .
COPY models/ ./models/

# Expose the application port
EXPOSE 5000

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Start the Flask app using Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]
