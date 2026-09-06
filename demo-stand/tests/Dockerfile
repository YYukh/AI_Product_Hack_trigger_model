# Образ для прогона тестов стенда и ml/common. Контекст сборки — корень репозитория.
#   docker build -f tests/Dockerfile -t stand-tests .
#   docker run --rm -v "$PWD:/w" stand-tests
FROM python:3.11-slim
WORKDIR /w
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/w:/w/ml:/w/ml/upstream \
    MLDATA_DIR=/tmp/mldata \
    CBR_FALLBACK_CSV=/w/ml/fallback/cbr_rates.csv

COPY ml/requirements-service.txt /tmp/base.txt
COPY tests/requirements-test.txt /tmp/test.txt
RUN pip install --no-cache-dir -r /tmp/base.txt -r /tmp/test.txt

CMD ["python", "-m", "pytest", "tests", "-q"]
