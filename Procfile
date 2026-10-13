# Used by Render / Railway / Heroku-style platforms to start the app.
# gunicorn is a production WSGI server; Flask's own dev server is not meant to face the
# internet. `--chdir part4_web` because app.py lives there and expects to import its
# sibling part1_assistant package. One worker with threads suits this app: the model and
# the BM25 index are loaded once per worker, and the work is waiting on an API, not CPU.
web: gunicorn --chdir part4_web app:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:$PORT
