import logging
from pyspark.sql import SparkSession
from config.app_config import SPARK_CONFIG

def create_spark_session() -> SparkSession:
    """Creates a SparkSession using settings from the config file."""
    spark = SparkSession.builder \
        .appName(SPARK_CONFIG["app_name"]) \
        .master(SPARK_CONFIG["master"]) \
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
        .getOrCreate()
    return spark

def setup_logging():
    """Configures the logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )