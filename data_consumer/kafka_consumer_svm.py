# from kafka import KafkaConsumer
# import json
# import logging
# from pymongo import MongoClient
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, udf, array
# from pyspark.ml import PipelineModel
# from pyspark.sql.types import ArrayType, FloatType, StringType, DoubleType
# from pyspark.ml.functions import vector_to_array
# from pyspark import SparkConf
# import numpy as np
# import re
# import os

# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# kafka_server = 'kafka:9092'
# mongo_host = 'host.docker.internal'  # Connect to MongoDB running on the host machine

# conf = SparkConf() \
#     .set("spark.executor.memory", "4g") \
#     .set("spark.driver.memory", "4g") \
#     .set("spark.network.timeout", "600s") \
#     .set("spark.executor.heartbeatInterval", "60s")

# spark = SparkSession.builder \
#     .appName("KafkaStreamWithMLPredictions") \
#     .config(conf=conf) \
#     .getOrCreate()

# # Load pre-trained svm model
# svm_model_path = "/app/data_consumer/svm_model"
# svm_model = PipelineModel.load(svm_model_path)
# logger.info('load model success')

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
#     consumer_timeout_ms=10000,
#     value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# # Dictionary to map numeric labels to sentiment text
# label_to_sentiment = {0: "Neutral", 1: "Positive", 2: "Negative", 3: "Irrelevant"}
# sentiment_mapping_udf = udf(lambda x: label_to_sentiment[x], StringType())

# # Clean the content by removing non-alphabetic characters and converting to lowercase
# def clean_content(content):
#     cleaned_content = re.sub(r'[^a-zA-Z\s]', '', content)
#     cleaned_content = cleaned_content.lower()
#     return cleaned_content

# # Define the softmax function
# def softmax(raw_predictions):
#     exps = np.exp(raw_predictions)
#     return (exps / exps.sum()).tolist()

# # Register the softmax UDF
# softmax_udf = udf(softmax, ArrayType(FloatType()))

# # Define a UDF to get the softmax value at the index of svm_prediction
# def get_softmax_at_index(softmax_values, index):
#     return float(softmax_values[index])

# get_softmax_at_index_udf = udf(get_softmax_at_index, DoubleType())
# for message in consumer:
#     try:
#         data = message.value
#         logger.info(f"Received message: {data}")
#         # data.update({"Cleaned Content":clean_content(data['Content'])})
#         df = spark.createDataFrame([data])
#         predictions = svm_model.transform(df)
#         logger.info("predict success")
#         predictions = predictions.withColumn('Softmax', softmax_udf(vector_to_array(predictions['rawPrediction'])))
#         predictions = predictions.withColumn('Confidence Score', get_softmax_at_index_udf(col('Softmax'), col('svm_prediction').cast("int")))
#         predictions = predictions.withColumn('Predicted Sentiment', sentiment_mapping_udf(predictions['svm_prediction']))
        
#         # Select and rename columns
#         predicted_data = predictions.select(
#             col('ID'),
#             col('Entity'),
#             col('Content'),
#             col('Predicted Sentiment').alias('Predicted_Sentiment'),
#             col('Confidence Score').alias('Confidence_Score')
#         )
        
#         # Convert the DataFrame to a dictionary
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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, from_json
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark import SparkConf
import numpy as np
from pymongo import MongoClient
import logging
import re
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kafka_server = 'kafka:9092'
mongo_host = 'host.docker.internal'  # Connect to MongoDB running on the host machine

conf = SparkConf() \
    .set("spark.executor.memory", "4g") \
    .set("spark.driver.memory", "4g") \
    .set("spark.network.timeout", "600s") \
    .set("spark.executor.heartbeatInterval", "60s")

spark = SparkSession.builder \
    .appName("KafkaStreamWithMLPredictions") \
    .config(conf=conf) \
    .getOrCreate()

# Load pre-trained SVM model
svm_model_path = "/app/data_consumer/svm_model"
svm_model = PipelineModel.load(svm_model_path)
logger.info('load model success')

# Connect to MongoDB
mongo_client = MongoClient(f'mongodb://{mongo_host}:27017')
db = mongo_client['fb_db']
collection = db['fb_collection']

# Define schema for Kafka messages
schema = StructType([
    StructField("ID", StringType(), True),
    StructField("Entity", StringType(), True),
    StructField("Content", StringType(), True),
    StructField("Cleaned Content", StringType(), True)
])

# Dictionary to map numeric labels to sentiment text
label_to_sentiment = {0: "Neutral", 1: "Positive", 2: "Negative", 3: "Irrelevant"}
sentiment_mapping_udf = udf(lambda x: label_to_sentiment[x], StringType())

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

# Define the Kafka stream
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", "fb_data") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse the JSON messages
parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Apply the pre-trained model to the stream
predictions = svm_model.transform(parsed_stream)

# Convert raw predictions to softmax probabilities and calculate confidence score
predictions = predictions.withColumn('Softmax', softmax_udf(vector_to_array(predictions['rawPrediction'])))
predictions = predictions.withColumn('Confidence Score', get_softmax_at_index_udf(col('Softmax'), col('svm_prediction').cast("int")))
predictions = predictions.withColumn('Predicted Sentiment', sentiment_mapping_udf(predictions['svm_prediction']))

# Select necessary columns for storage
predicted_data = predictions.select(
    col('ID').alias('ID'),
    col('Entity'),
    col('Content'),
    col('Predicted Sentiment').alias('Predicted_Sentiment'),
    col('Confidence Score').alias('Confidence_Score')
)

# Define a function to save the results to MongoDB
def save_to_mongo(df, epoch_id):
    predicted_data_dict = [row.asDict() for row in df.collect()]
    if predicted_data_dict:
        collection.insert_many(predicted_data_dict)
        logger.info("Predicted data stored in MongoDB")
    else:
        logger.info("No data to store")

# Write the stream to MongoDB
query = predicted_data.writeStream \
    .foreachBatch(save_to_mongo) \
    .outputMode("append") \
    .start()

# Await termination
query.awaitTermination()

# Cleanup Spark session
spark.stop()
