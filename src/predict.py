import os
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import OneVsRestModel, LogisticRegressionModel, NaiveBayesModel
from pyspark.ml.feature import IndexToString  

# Import utility functions and config variables
from src.utils import create_spark_session, setup_logging
from config.app_config import PIPELINE_PATH, ENSEMBLE_MODEL_PATHS,Conda_env_path
from pyspark.sql import functions as F

os.environ['PYSPARK_PYTHON'] = Conda_env_path
os.environ['PYSPARK_DRIVER_PYTHON'] = Conda_env_path

class EnsemblePredictor:
    """Loads a saved ensemble model and makes predictions on new data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_models()

    def _load_models(self):
        """Loads the feature pipeline and all three base models for the ensemble."""
        self.logger.info("Loading feature engineering pipeline...")
        self.pipeline = PipelineModel.load(PIPELINE_PATH)

        self.logger.info("Loading ensemble base models...")
        self.gbt_model = OneVsRestModel.load(ENSEMBLE_MODEL_PATHS['gbt'])
        self.lr_model = LogisticRegressionModel.load(ENSEMBLE_MODEL_PATHS['lr'])
        self.nb_model = NaiveBayesModel.load(ENSEMBLE_MODEL_PATHS['nb'])
        self.logger.info("All models loaded successfully.")

    def predict(self, new_data: DataFrame) -> DataFrame:
        """
        Makes a final prediction using the voting ensemble.

        Args:
            new_data (DataFrame): A Spark DataFrame with a 'lyrics' column.

        Returns:
            DataFrame: The input DataFrame with an added human-readable 'genre_prediction' column.
        """
        # 1. Apply the feature engineering pipeline
        self.logger.info("Applying feature transformations...")
        transformed_data = self.pipeline.transform(new_data)

        # 2. Get predictions from each base model
        self.logger.info("Generating predictions from each base model...")
        preds_gbt = self.gbt_model.transform(transformed_data).withColumnRenamed("prediction", "gbt_pred")
        preds_lr = self.lr_model.transform(transformed_data).withColumnRenamed("prediction", "lr_pred")
        preds_nb = self.nb_model.transform(transformed_data).withColumnRenamed("prediction", "nb_pred")
        
        # 3. Join predictions
        preds_gbt = preds_gbt.withColumn("id", F.monotonically_increasing_id())
        preds_lr = preds_lr.withColumn("id", F.monotonically_increasing_id())
        preds_nb = preds_nb.withColumn("id", F.monotonically_increasing_id())

        combined_preds = preds_gbt.select("id", "lyrics", "gbt_pred") \
            .join(preds_lr.select("id", "lr_pred"), "id") \
            .join(preds_nb.select("id", "nb_pred"), "id")

        # 4. Apply voting logic
        def majority_vote(*predictions):
            from collections import Counter
            return Counter(predictions).most_common(1)[0][0]

        vote_udf = udf(majority_vote, DoubleType())

        final_preds_indexed = combined_preds.withColumn("prediction", vote_udf(col("gbt_pred"), col("lr_pred"), col("nb_pred")))
        
        # --- NEW: Convert prediction index back to genre string ---
        self.logger.info("Converting predicted labels back to genre names...")
        string_indexer_model = self.pipeline.stages[0] 

        label_converter = IndexToString(
            inputCol="prediction",
            outputCol="genre_prediction",
            labels=string_indexer_model.labels 
        )

        final_predictions = label_converter.transform(final_preds_indexed)
        return final_predictions.select("lyrics", "genre_prediction")

if __name__ == '__main__':
    setup_logging()
    spark = create_spark_session()

    sample_lyrics = [
        ("love love love peace and harmony makes the world go round",),
        ("whiskey blues and cheap guitars thats the life for me",),
        ("yeah microphone check one two this is how we do it",)
    ]
    new_songs_df = spark.createDataFrame(sample_lyrics, ["lyrics"])
    
    # Create a predictor and make predictions
    predictor = EnsemblePredictor()
    predictions_df = predictor.predict(new_songs_df)
    
    logging.info("--- Ensemble Predictions ---")
    predictions_df.show(truncate=False)
    
    spark.stop()