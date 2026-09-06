"""Конфигурация стенда. Всё — из переменных окружения, значения по умолчанию
совпадают с .env.example и с решениями user-path-v2 / ТЗ."""
import os

VERSION = "0.2.0"

# Пути к данным (том ./data:/data:ro в compose; локально — ../data).
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))

# Источник сигналов: пусто → файл data/signals.json; иначе — HTTP к модели пуша.
ML_URL = os.getenv("ML_URL", "").strip()
# Источник курсов: пусто → файл data/rates.csv; иначе — HTTP к парсеру котировок.
RATES_URL = os.getenv("RATES_URL", "").strip()
# Сырые сигналы движков «выгодного момента» для экрана данных (только проксирование).
MOMENT_URL = os.getenv("MOMENT_URL", "").strip()
ML_TIMEOUT_S = float(os.getenv("ML_TIMEOUT_S", "2"))

# Порог состояния DRIFT: дельта против клиента, базисные пункты.
DRIFT_THRESHOLD_BP = int(os.getenv("DRIFT_THRESHOLD_BP", "20"))

# Срок жизни пуша: после него открытие ведёт на нейтральный экран.
PUSH_TTL_MINUTES = int(os.getenv("PUSH_TTL_MINUTES", str(24 * 60)))

# Коммуникационная политика.
PUSH_BUDGET_MONTH = int(os.getenv("PUSH_BUDGET_MONTH", "8"))
PUSH_COOLDOWN_DAYS = int(os.getenv("PUSH_COOLDOWN_DAYS", "3"))
QUIET_HOURS = (22, 9)  # 22:00–09:00 по таймзоне портрета

# Резервирование.
FEATURE_RESERVE = os.getenv("FEATURE_RESERVE", "true").lower() == "true"
RESERVE_TTL_DAYS = int(os.getenv("RESERVE_TTL_DAYS", "7"))
RESERVE_PERCENTILE = int(os.getenv("RESERVE_PERCENTILE", "25"))
RESERVE_WINDOW_DAYS = int(os.getenv("RESERVE_WINDOW_DAYS", "30"))

# Право вето получателя (опциональный сюжет S7 / O1).
FEATURE_RECIPIENT_LIMIT = os.getenv("FEATURE_RECIPIENT_LIMIT", "true").lower() == "true"
