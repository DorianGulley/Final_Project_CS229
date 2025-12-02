FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps for packages that may need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Default entrypoint — run training CLI
ENTRYPOINT ["python", "train.py"]
