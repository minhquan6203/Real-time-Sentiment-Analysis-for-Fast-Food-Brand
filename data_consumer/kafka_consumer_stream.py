from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array
from pyspark.ml import PipelineModel
from pyspark.sql.types import ArrayType, FloatType, StringType, DoubleType
from pyspark.ml.functions import vector_to_array
import re
import os
import numpy as np

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaStreamWithMLPredictions") \
    .getOrCreate()

# Load pre-trained svm model
svm_model_path = "/app/data_consumer/svm_model"
svm_model = PipelineModel.load(svm_model_path)

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

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "fb_data") \
    .option("startingOffsets", "earliest") \
    .load()

# Convert the value column from binary to string
df = df.selectExpr("CAST(value AS STRING) as value")

# Parse the JSON content
df = df.selectExpr("json_tuple(value, 'ID', 'Entity', 'Content') as (ID, Entity, Content)")

# Clean content
clean_content_udf = udf(clean_content, StringType())
df = df.withColumn("Cleaned Content", clean_content_udf(col("Content")))

# Make predictions
predictions = svm_model.transform(df)
predictions = predictions.withColumn('Softmax', softmax_udf(vector_to_array(predictions['rawPrediction'])))
predictions = predictions.withColumn('Confidence_Score', get_softmax_at_index_udf(col('Softmax'), col('svm_prediction').cast("int")))
predictions = predictions.withColumn('Predicted_Sentiment', sentiment_mapping_udf(predictions['svm_prediction']))

# Select and rename columns
predicted_data = predictions.select(
    col('ID').alias('ID'),
    col('Entity'),
    col('Content'),
    col('Predicted_Sentiment').alias('Predicted_Sentiment'),
    col('Confidence_Score').alias('Confidence_Score')
)

# Write the predictions to MongoDB
predicted_data.writeStream \
    .format("com.mongodb.spark.sql.DefaultSource") \
    .option("spark.mongodb.output.uri", "mongodb://host.docker.internal:27017/fb_db.fb_collection") \
    .outputMode("append") \
    .start() \
    .awaitTermination()
