FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY backend/services /app/backend/services
COPY pipelines/training /app/pipelines/training

WORKDIR /app/backend/services

EXPOSE 8000

CMD ["uvicorn", "scheduleServer:app", "--host", "0.0.0.0", "--port", "8000"]
