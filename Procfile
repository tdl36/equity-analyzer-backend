web: gunicorn app_v3:app --workers 1 --threads 4 --worker-class gthread --timeout 300 --bind 0.0.0.0:$PORT --log-level info --error-logfile - --access-logfile -
