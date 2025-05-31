FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY . .

EXPOSE 8000
ENTRYPOINT ["python", "main.py"]
