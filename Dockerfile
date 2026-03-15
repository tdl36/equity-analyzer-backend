FROM python:3.11-slim

# Install ffmpeg for audio chunking/splitting
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD gunicorn app_v3:app --config gunicorn.conf.py --bind 0.0.0.0:$PORT --log-level info --error-logfile - --access-logfile -
