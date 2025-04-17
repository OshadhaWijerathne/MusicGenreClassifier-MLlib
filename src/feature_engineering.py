# src/feature_engineering.py

import os
import logging
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer

# Import utility functions and config variables
from src.utils import create_spark_session, setup_logging
from config.app_config import TRAIN_DATA_PATH, PIPELINE_PATH, FEATURE_CONFIG

class FeaturePipelineBuilder:
    """Class for creating and managing the feature engineering pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _create_label_indexer(self) -> StringIndexer:
        """Creates the StringIndexer stage for the target variable."""
        self.logger.info("Creating label indexer stage.")
        return StringIndexer(inputCol="genre", outputCol="label")

    def _create_tokenizer(self) -> Tokenizer:
        """Creates the Tokenizer stage for the lyrics."""
        self.logger.info("Creating tokenizer stage.")
        return Tokenizer(inputCol="lyrics", outputCol="words")

    def _create_stopwords_remover(self) -> StopWordsRemover:
        """Creates the StopWordsRemover stage."""
        self.logger.info("Creating stop words remover stage.")
        return StopWordsRemover(inputCol="words", outputCol="filtered_words")

    def _create_hashing_tf(self) -> HashingTF:
        """Creates the HashingTF stage using parameters from config."""
        self.logger.info("Creating HashingTF stage.")
        return HashingTF(
            inputCol="filtered_words", 
            outputCol="raw_features", 
            numFeatures=FEATURE_CONFIG["hashing_tf_features"] 
        )

    def _create_idf(self) -> IDF:
        """Creates the IDF stage for inverse document frequency."""
        self.logger.info("Creating IDF stage.")
        return IDF(inputCol="raw_features", outputCol="features")

    def build(self) -> Pipeline:
        """Assembles and returns the full feature engineering pipeline."""
        self.logger.info("Assembling all stages into a final pipeline.")
        stages = [
            self._create_label_indexer(),
            self._create_tokenizer(),
            self._create_stopwords_remover(),
            self._create_hashing_tf(),
            self._create_idf()
        ]
        return Pipeline(stages=stages)

def run_feature_engineering(spark):
    """Executes the full feature engineering workflow."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Feature Engineering Workflow ---")
    
    logger.info(f"Loading training data from {TRAIN_DATA_PATH}")
    df_train = spark.read.csv(TRAIN_DATA_PATH, header=True, inferSchema=True)

    builder = FeaturePipelineBuilder()
    feature_pipeline = builder.build()

    logger.info("Fitting the pipeline on the training data...")
    pipeline_model = feature_pipeline.fit(df_train)

    logger.info(f"Saving the fitted pipeline to {PIPELINE_PATH}")
    pipeline_model.write().overwrite().save(PIPELINE_PATH)
    
    logger.info("--- Feature Engineering Workflow Complete ---")


if __name__ == "__main__":
    setup_logging()
    
    spark_session = create_spark_session()
    run_feature_engineering(spark_session)
    spark_session.stop()