from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType, IntegerType, DateType
from pyspark.ml import PipelineModel
from pymongo import MongoClient
import logging
import os
import re
from datetime import datetime, date

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka and MongoDB configuration
kafka_server = 'kafka:9092'
mongo_host = 'host.docker.internal'

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaStreamWithMLPredictions") \
    .getOrCreate()

# Load pre-trained Logistic Regression model
lr_model_path = "/app/data_consumer/lr_model"
lr_model = PipelineModel.load(lr_model_path)

# Connect to MongoDB
mongo_client = MongoClient(f'mongodb://{mongo_host}:27017')
db = mongo_client['fb_db']
collection = db['fb_collection']

# Define schema for Kafka messages
schema = StructType([
    StructField("ID", StringType(), True),
    StructField("Entity", StringType(), True),
    StructField("Content", StringType(), True),
    StructField("Cleaned Content", StringType(), True),
    StructField("ID_Commentt", StringType(), True),
    StructField("ID_User", StringType(), True),
    StructField("Comment_Like", IntegerType(), True),
    StructField("Date", DateType(), True)
])

# UDF to convert DenseVector to list and calculate max probability
def dense_vector_to_list(vector):
    probabilities = vector.toArray().tolist()
    max_probability = max(probabilities)
    return probabilities, max_probability

convert_udf = udf(dense_vector_to_list, ArrayType(FloatType()))

# Dictionary to map numeric labels to sentiment text
label_to_sentiment = {0: "Neutral", 1: "Positive", 2: "Negative", 3: "Irrelevant"}
sentiment_mapping_udf = udf(lambda x: label_to_sentiment[x], StringType())

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
predictions = lr_model.transform(parsed_stream)

# Convert DenseVector to list and find max probability
predictions = predictions.withColumn('Confidence', convert_udf(predictions['probability']))
predictions = predictions.withColumn('Confidence_Score', predictions['Confidence'][1] * 100)

# Map numeric prediction to sentiment text
predictions = predictions.withColumn('Predicted_Sentiment', sentiment_mapping_udf(predictions['prediction']))

# Select necessary columns for storage
predicted_data = predictions.select(
    col('ID'),
    col('Entity'),
    col('Content'),
    col('Predicted_Sentiment'),
    col('Confidence_Score'),
    col('ID_Commentt'),
    col('ID_User'),
    col('Comment_Like'),
    # col('Date')
)

# Function to convert date to datetime
# def date_to_datetime(date):
#     if isinstance(date, datetime):
#         return date
#     if isinstance(date, date):
#         return datetime.combine(date, datetime.min.time())
#     return date

# Define a function to save the results to MongoDB
def save_to_mongo(df, epoch_id):
    predicted_data_dict = [row.asDict() for row in df.collect()]
    
    # Convert date to datetime
    # for data in predicted_data_dict:
    #     if 'Date' in data:
    #         data['Date'] = date_to_datetime(data['Date'])
    
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