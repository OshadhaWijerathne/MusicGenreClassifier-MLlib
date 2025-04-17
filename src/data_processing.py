# src/data_processing.py

import os
import logging
from pyspark.sql import DataFrame

# Import utility functions and config variables
from src.utils import create_spark_session, setup_logging
from config.app_config import RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURE_CONFIG

class DataProcessor:
    """Handles loading, cleaning, splitting, and saving of the dataset using centralized config."""
    
    def __init__(self, spark):
        """
        Initializes the DataProcessor.

        Args:
            spark: The SparkSession object.
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def _load_data(self) -> DataFrame:
        """Loads the raw dataset from the path specified in the config."""
        self.logger.info(f"Loading raw data from {RAW_DATA_PATH}")
        return self.spark.read.csv(RAW_DATA_PATH, header=True, inferSchema=True)

    def _clean_data(self, df: DataFrame) -> DataFrame:
        """Cleans the DataFrame by removing duplicates and nulls."""
        self.logger.info("Cleaning data: dropping duplicates, selecting columns, and handling nulls.")
        df_cleaned = df.dropDuplicates()
        df_cleaned = df_cleaned.select("lyrics", "genre").dropna(subset=["lyrics", "genre"])
        self.logger.info(f"Total records after cleaning: {df_cleaned.count()}")
        return df_cleaned

    def _split_and_save_data(self, df: DataFrame):
        """Splits data and saves it to the paths specified in the config."""
        self.logger.info("Splitting data into 80/20 train/test sets.")
        (train_df, test_df) = df.randomSplit(
            [0.8, 0.2], 
            seed=FEATURE_CONFIG["random_split_seed"]
        )

        self.logger.info(f"Training data count: {train_df.count()}")
        self.logger.info(f"Test data count: {test_df.count()}")

        self.logger.info(f"Saving training data to {TRAIN_DATA_PATH}")
        train_df.write.mode("overwrite").csv(TRAIN_DATA_PATH, header=True)
        
        self.logger.info(f"Saving test data to {TEST_DATA_PATH}")
        test_df.write.mode("overwrite").csv(TEST_DATA_PATH, header=True)

    def run(self):
        """Executes the full data processing workflow."""
        self.logger.info("--- Starting Data Processing Workflow ---")
        raw_df = self._load_data()
        cleaned_df = self._clean_data(raw_df)
        self._split_and_save_data(cleaned_df)
        self.logger.info("--- Data Processing Workflow Complete ---")


if __name__ == "__main__":
    setup_logging()

    spark_session = create_spark_session()
    processor = DataProcessor(spark_session)
    processor.run()
    spark_session.stop()