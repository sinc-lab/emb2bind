FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

COPY requirements-container.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
 && pip install -r requirements-container.txt

COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY predict.py /app/

ENTRYPOINT ["python", "/app/predict.py", \
            "--device", "cpu", \
            "--fasta", "/data/input.fasta", \
            "--embedding-dir", "/data/embeddings", \
            "--output-dir", "/output", \
            "--threads", "4"]
CMD []