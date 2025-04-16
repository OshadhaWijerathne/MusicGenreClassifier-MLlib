# src/utils.py

import logging
from pyspark.sql import SparkSession

def create_spark_session(app_name: str) -> SparkSession:
    """
    Creates and returns a pre-configured SparkSession.

    Args:
        app_name (str): The name for the Spark application.

    Returns:
        SparkSession: The configured Spark session object.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

def setup_logging():
    """Configures the logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )