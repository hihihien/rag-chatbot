# backend/Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn serve_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
