from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import csv
import json
import logging
from time import sleep
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kafka_server = 'kafka:9092'

# Setup Kafka Admin Client to manage topics
admin_client = KafkaAdminClient(
    bootstrap_servers=kafka_server,
    client_id='fb_producer'
)

topic_name = "fb_data"
# Check if the topic already exists and create it if not
try:
    existing_topics = admin_client.list_topics()
    if topic_name not in existing_topics:
        topic = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
        admin_client.create_topics(new_topics=[topic])
        logger.info(f"Created new topic: {topic_name}")
    else:
        logger.info(f"Topic '{topic_name}' already exists.")
except Exception as e:
    logger.error(f"Failed to create topic: {e}")
    exit(1)  # Exit if topic creation fails

# Setup Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=kafka_server,
    api_version=(0, 11, 5),
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def clean_content(content):
    cleaned_content = re.sub(r'[^a-zA-Z\s]', '', content)
    cleaned_content = cleaned_content.lower()
    return cleaned_content

with open('all_comments.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)
    random.shuffle(data)
    sleep(90)
    for row in data:
        message = {
            "ID": int(row['ID']),
            "Entity": row['Entity'],
            # "Sentiment": row['Sentiment'],
            "Content": row['Content'],
            "Cleaned Content": clean_content(row['Content']),
            "ID_Comment": row['ID_Comment'],
            "ID_User": row['ID_User'],
            "Comment_Like": row['Comment_Like'],
            "Date": row['Date']
        }
        producer.send(topic_name, value=message)
        # Uncomment the following line to log every message sent (can generate lots of logs)
        logger.info(f"Produced message comment of {row['Entity']}")
        sleep(0.5)

# Ensure all messages are sent
producer.flush()