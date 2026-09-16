FROM python:3.11-slim

WORKDIR /app

# LightGBM needs libgomp at runtime; curl is used by the healthcheck;
# supervisor runs the Streamlit UI and the FastAPI REST API as two processes
# in this one container (see supervisord.conf).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 8000

# Checks the Streamlit UI specifically — retrain/one-off CLI commands
# (docker-compose run --entrypoint python app -m src.ml.train) bypass the
# ENTRYPOINT entirely and are unaffected by this.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["supervisord", "-c", "/app/supervisord.conf"]
