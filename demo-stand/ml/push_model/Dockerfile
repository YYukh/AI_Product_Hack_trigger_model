# build context: ./ml
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/upstream \
    MLDATA_DIR=/mldata \
    CBR_FALLBACK_CSV=/app/fallback/cbr_rates.csv

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY upstream/src ./upstream/src
COPY common ./common
COPY fallback ./fallback
COPY push_model/app.py ./app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
