FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY arl_eegmodels ./arl_eegmodels
COPY eeg_project ./eeg_project
COPY inference.py api.py ./
COPY artifacts ./artifacts
ENV EEG_MIXED_ENSEMBLE_MANIFEST=artifacts/production/manifest.json
ENV EEG_REQUIRE_MODEL=true
ENV EEG_VERIFY_ARTIFACTS=true
ENV EEG_JSON_LOGS=true
ENV EEG_MAX_CONCURRENT_INFERENCES=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
