# FROM python:3.7

# COPY requirements.txt /requirements.txt
# RUN pip install -r requirements.txt

# WORKDIR /app
# COPY . /app
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl --fail http://localhost:8000 || exit 1
FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl --fail http://localhost:80 || exit 1