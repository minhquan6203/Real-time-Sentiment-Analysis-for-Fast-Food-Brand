from kafka import KafkaConsumer
import json
import logging
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array
from pyspark.ml import PipelineModel
from pyspark.sql.types import ArrayType, FloatType, StringType, DoubleType
from pyspark.ml.functions import vector_to_array
import numpy as np
import re
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kafka_server = 'kafka:9092'
mongo_host = 'host.docker.internal'  # Connect to MongoDB running on the host machine

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaStreamWithMLPredictions") \
    .getOrCreate()

# Load pre-trained svm model
svm_model_path = "/app/data_consumer/svm_model"
svm_model = PipelineModel.load(svm_model_path)

# Connect to MongoDB using host.docker.internal
mongo_client = MongoClient(f'mongodb://{mongo_host}:27017')
db = mongo_client['fb_db']
collection = db['fb_collection']

# Define Kafka consumer
consumer = KafkaConsumer(
    'fb_data',
    bootstrap_servers=kafka_server,
    auto_offset_reset='earliest',
    api_version=(0, 11, 5),
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Dictionary to map numeric labels to sentiment text
label_to_sentiment = {0: "Neutral", 1: "Positive", 2: "Negative", 3: "Irrelevant"}
sentiment_mapping_udf = udf(lambda x: label_to_sentiment[x], StringType())

# Clean the content by removing non-alphabetic characters and converting to lowercase
def clean_content(content):
    cleaned_content = re.sub(r'[^a-zA-Z\s]', '', content)
    cleaned_content = cleaned_content.lower()
    return cleaned_content

# Define the softmax function
def softmax(raw_predictions):
    exps = np.exp(raw_predictions)
    return (exps / exps.sum()).tolist()

# Register the softmax UDF
softmax_udf = udf(softmax, ArrayType(FloatType()))

# Define a UDF to get the softmax value at the index of svm_prediction
def get_softmax_at_index(softmax_values, index):
    return float(softmax_values[index])

get_softmax_at_index_udf = udf(get_softmax_at_index, DoubleType())
for message in consumer:
    try:
        data = message.value
        logger.info(f"Received message: {data}")

        data.update({"Cleaned Content":clean_content(data['Content'])})
        df = spark.createDataFrame([data])
        predictions = svm_model.transform(df)
        predictions = predictions.withColumn('Softmax', softmax_udf(vector_to_array(predictions['rawPrediction'])))
        predictions = predictions.withColumn('Confidence Score', get_softmax_at_index_udf(col('Softmax'), col('svm_prediction').cast("int")))
        predictions = predictions.withColumn('Predicted Sentiment', sentiment_mapping_udf(predictions['svm_prediction']))
        
        # Select and rename columns
        predicted_data = predictions.select(
            col('ID').alias('ID'),
            col('Entity'),
            col('Content'),
            col('Predicted Sentiment').alias('Predicted_Sentiment'),
            col('Confidence Score').alias('Confidence_Score')
        )
        
        # Convert the DataFrame to a dictionary
        predicted_data_dict = [row.asDict() for row in predicted_data.collect()]

        # Store the predicted message in MongoDB
        if predicted_data_dict:
            collection.insert_many(predicted_data_dict)
            logger.info("Predicted data stored in MongoDB")
        else:
            logger.info("No data to store")

    except Exception as e:
        logger.error(f"Error processing message: {e}")

# Cleanup Spark session
spark.stop()

# from kafka import KafkaConsumer
# import json
# import logging
# from pymongo import MongoClient
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when, udf
# from pyspark.sql.types import StringType, FloatType
# import random
# import os

# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# kafka_server = 'kafka:9092'
# mongo_host = 'host.docker.internal'  # Connect to MongoDB running on the host machine

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("KafkaStreamWithRandomPredictions") \
#     .getOrCreate()

# # Connect to MongoDB using host.docker.internal
# mongo_client = MongoClient(f'mongodb://{mongo_host}:27017')
# db = mongo_client['fb_db']
# collection = db['fb_collection']

# # Define Kafka consumer
# consumer = KafkaConsumer(
#     'fb_data',
#     bootstrap_servers=kafka_server,
#     auto_offset_reset='earliest',
#     api_version=(0, 11, 5),
#     enable_auto_commit=True,
#     value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# # Dictionary to map numeric labels to sentiment text
# label_to_sentiment = {0: "Neutral", 1: "Positive", 2: "Negative", 3: "Irrelevant"}
# sentiment_mapping_udf = udf(lambda x: label_to_sentiment[int(x)], StringType())

# # UDF to generate random predictions
# def random_predict():
#     return random.randint(0, 3)

# random_predict_udf = udf(random_predict, StringType())

# # UDF to generate random accuracy
# def random_accuracy():
#     return random.uniform(50, 100)

# random_accuracy_udf = udf(random_accuracy, FloatType())

# for message in consumer:
#     try:
#         data = message.value
#         logger.info(f"Received message: {data}")

#         # Adjust column names to match the model's expected input
#         data['content'] = data.pop('Content', None)

#         # Create a DataFrame
#         df = spark.createDataFrame([data])

#         # Add label column for sentiment
#         df = df.withColumn('label', when(col('Sentiment') == 'Positive', 1)
#                                    .when(col('Sentiment') == 'Negative', 2)
#                                    .when(col('Sentiment') == 'Neutral', 0)
#                                    .when(col('Sentiment') == 'Irrelevant', 3))

#         # Make random predictions
#         predictions = df.withColumn('prediction', random_predict_udf())

#         # Generate random accuracy
#         predictions = predictions.withColumn('Accuracy', random_accuracy_udf())

#         # Map numeric prediction to sentiment text
#         predictions = predictions.withColumn('Predicted Sentiment', sentiment_mapping_udf(predictions['prediction']))

#         # Select necessary columns for storage with adjusted column names
#         predicted_data = predictions.select(
#             col('ID').alias('ID'),
#             col('Entity'),
#             col('content').alias('Content'),
#             col('Predicted Sentiment').alias('Predicted_Sentiment'),
#             col('Accuracy')
#         )

#         # Convert DataFrame to list of dictionaries to insert into MongoDB
#         predicted_data_dict = [row.asDict() for row in predicted_data.collect()]

#         # Store the predicted message in MongoDB
#         if predicted_data_dict:
#             collection.insert_many(predicted_data_dict)
#             logger.info("Predicted data stored in MongoDB")
#         else:
#             logger.info("No data to store")

#     except Exception as e:
#         logger.error(f"Error processing message: {e}")

# # Cleanup Spark session
# spark.stop()
