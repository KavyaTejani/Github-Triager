FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install -e ".[inference,redis]"
COPY . .
EXPOSE 8000
# REDIS_URL can be injected via docker-compose or HF Space secrets
ENV REDIS_URL=""
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
