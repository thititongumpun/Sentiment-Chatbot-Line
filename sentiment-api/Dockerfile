FROM python:3.8-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install fastapi uvicorn
RUN pip cache purge
RUN pip install -r requirements.txt

# RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install curl -y

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl --fail http://localhost:80 || exit 1