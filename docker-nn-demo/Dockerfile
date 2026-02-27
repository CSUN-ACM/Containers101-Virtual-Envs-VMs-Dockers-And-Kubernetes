FROM python:3.11-slim

WORKDIR /app

# Copy dependencies first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY config.yaml .
COPY src/ ./src/

CMD ["python", "src/train.py"]
