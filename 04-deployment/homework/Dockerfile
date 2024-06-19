FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "starter.py", "yellow","2023","5"]
