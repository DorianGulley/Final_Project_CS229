FROM python:3.11-slim

# Install system deps + OpenBLAS (optimized BLAS library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment to use OpenBLAS
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV OMP_NUM_THREADS=4

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /app
ENTRYPOINT ["python", "train.py"]