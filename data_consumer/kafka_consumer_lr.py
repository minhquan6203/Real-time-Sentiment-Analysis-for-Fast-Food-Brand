# from kafka import KafkaConsumer
# import json
# import logging
# from pymongo import MongoClient
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when, udf
# from pyspark.ml import PipelineModel
# from pyspark.sql.types import ArrayType, FloatType, StringType
# from pyspark.ml.linalg import DenseVector
# import os
# import re
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# kafka_server = 'kafka:9092'
# mongo_host = 'host.docker.internal'  # Connect to MongoDB running on the host machine

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("KafkaStreamWithMLPredictions") \
#     .getOrCreate()

# # Load pre-trained Logistic Regression model
# lr_model_path = "/app/data_consumer/lr_model"
# lr_model = PipelineModel.load(lr_model_path)

# # Connect to MongoDB using host.docker.internal
# mongo_client = MongoClient(f'mongodb://{mongo_host}:27017')
# db = mongo_client['fb_db']
# collection = db['fb_collection']

# # UDF to convert DenseVector to list and calculate max probability
# def dense_vector_to_list(vector):
#     probabilities = vector.toArray().tolist()
#     max_probability = max(probabilities)
#     return probabilities, max_probability

# convert_udf = udf(dense_vector_to_list, ArrayType(FloatType()))

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
# sentiment_mapping_udf = udf(lambda x: label_to_sentiment[x], StringType())

# def clean_content(content):
#     cleaned_content = re.sub(r'[^a-zA-Z\s]', '', content)
#     cleaned_content = cleaned_content.lower()
#     return cleaned_content

# for message in consumer:
#     try:
#         data = message.value
#         logger.info(f"Received message: {data}")
#         # Create a DataFrame
#         df = spark.createDataFrame([data])
#         # Make predictions using the Logistic Regression model
#         predictions = lr_model.transform(df)


#         # Convert DenseVector to list and find max probability
#         predictions = predictions.withColumn('Confidence', convert_udf(predictions['probability']))
#         predictions = predictions.withColumn('Confidence_Score', predictions['Confidence'][1] * 100)

#         # Map numeric prediction to sentiment text
#         predictions = predictions.withColumn('Predicted Sentiment', sentiment_mapping_udf(predictions['prediction']))

#         # Select necessary columns for storage with adjusted column names
#         predicted_data = predictions.select(
#             col('ID'),
#             col('Entity'),
#             col('Content').alias('Content'),
#             col('Predicted Sentiment').alias('Predicted_Sentiment'),
#             col('Confidence_Score')
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



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType
from pyspark.ml import PipelineModel
from pymongo import MongoClient
import logging
import os
import re

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
    StructField("Cleaned Content", StringType(), True)
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
    col('Confidence_Score')
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
