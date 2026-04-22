FROM python:3.11-slim

WORKDIR /app

RUN groupadd -r namu && useradd -r -g namu namu

# gcc is required to compile uvloop and httptools (uvicorn[standard] extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R namu:namu /app

USER namu

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
