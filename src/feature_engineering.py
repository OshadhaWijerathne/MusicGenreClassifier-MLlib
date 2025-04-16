import os
import logging
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from utils import create_spark_session, setup_logging

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
        """Creates the HashingTF stage for term frequency."""
        self.logger.info("Creating HashingTF stage.")
        return HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

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

def run_feature_engineering(spark, train_path: str, pipeline_path: str):
    """
    Executes the full feature engineering workflow.
    
    Args:
        spark: The SparkSession object.
        train_path: Path to the cleaned training data.
        pipeline_path: Path to save the fitted pipeline model.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Feature Engineering Workflow ---")
    
    logger.info(f"Loading training data from {train_path}")
    df_train = spark.read.csv(train_path, header=True, inferSchema=True)

    builder = FeaturePipelineBuilder()
    feature_pipeline = builder.build()

    logger.info("Fitting the pipeline on the training data...")
    pipeline_model = feature_pipeline.fit(df_train)

    logger.info(f"Saving the fitted pipeline to {pipeline_path}")
    pipeline_model.write().overwrite().save(pipeline_path)
    
    logger.info("--- Feature Engineering Workflow Complete ---")


if __name__ == "__main__":
    setup_logging()
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_train.csv")
    PIPELINE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "feature_pipeline_lyrics_only")

    spark_session = create_spark_session("MusicClassifier_FeatureEngineering")
    
    run_feature_engineering(spark_session, TRAIN_DATA_PATH, PIPELINE_OUTPUT_PATH)

    spark_session.stop()