import os
import logging
from pyspark.sql import DataFrame
from utils import create_spark_session, setup_logging

class DataProcessor:
    """Handles loading, cleaning, splitting, and saving of the dataset."""
    
    def __init__(self, spark, input_path: str, train_path: str, test_path: str):
        """
        Initializes the DataProcessor.

        Args:
            spark: The SparkSession object.
            input_path: Path to the raw input CSV file.
            train_path: Path to save the cleaned training data.
            test_path: Path to save the cleaned test data.
        """
        self.spark = spark
        self.input_path = input_path
        self.train_path = train_path
        self.test_path = test_path
        self.logger = logging.getLogger(__name__)

    def _load_data(self) -> DataFrame:
        """Loads the raw dataset from the input path."""
        self.logger.info(f"Loading raw data from {self.input_path}")
        return self.spark.read.csv(self.input_path, header=True, inferSchema=True)

    def _clean_data(self, df: DataFrame) -> DataFrame:
        """Cleans the DataFrame by removing duplicates and nulls."""
        self.logger.info("Cleaning data: dropping duplicates, selecting columns, and handling nulls.")
        df_cleaned = df.dropDuplicates()
        df_cleaned = df_cleaned.select("lyrics", "genre").dropna(subset=["lyrics", "genre"])
        self.logger.info(f"Total records after cleaning: {df_cleaned.count()}")
        return df_cleaned

    def _split_and_save_data(self, df: DataFrame):
        """Splits data and saves it to the specified output paths."""
        self.logger.info("Splitting data into 80/20 train/test sets.")
        (train_df, test_df) = df.randomSplit([0.8, 0.2], seed=42)

        self.logger.info(f"Training data count: {train_df.count()}")
        self.logger.info(f"Test data count: {test_df.count()}")

        self.logger.info(f"Saving training data to {self.train_path}")
        train_df.write.mode("overwrite").csv(self.train_path, header=True)
        
        self.logger.info(f"Saving test data to {self.test_path}")
        test_df.write.mode("overwrite").csv(self.test_path, header=True)

    def run(self):
        """Executes the full data processing workflow."""
        self.logger.info("--- Starting Data Processing Workflow ---")
        raw_df = self._load_data()
        cleaned_df = self._clean_data(raw_df)
        self._split_and_save_data(cleaned_df)
        self.logger.info("--- Data Processing Workflow Complete ---")


if __name__ == "__main__":
    setup_logging()
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_dataset.csv")
    TRAIN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_train.csv")
    TEST_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_test.csv")

    spark_session = create_spark_session("MusicClassifier_DataProcessing")
    
    processor = DataProcessor(spark_session, RAW_DATA_PATH, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH)
    processor.run()
    
    spark_session.stop()