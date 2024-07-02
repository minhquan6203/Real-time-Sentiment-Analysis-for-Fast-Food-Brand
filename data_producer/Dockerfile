FROM python:3.8

WORKDIR /app

COPY kafka_producer.py /app
COPY requirements.txt /app
# COPY Dataset/twitter_training.csv /app
COPY Dataset/all_comments.csv /app
COPY Dataset/KFC.csv /app
COPY Dataset/DominosPizza.csv /app
COPY Dataset/Subway.csv app
COPY Dataset/McDonalds.csv app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "kafka_producer.py"]