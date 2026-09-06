FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    LOGS_DIR=/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
