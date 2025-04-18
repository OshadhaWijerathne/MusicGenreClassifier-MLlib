# app.py

import os
import logging
import json
from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import OneVsRestModel, LogisticRegressionModel, NaiveBayesModel
from pyspark.ml.feature import IndexToString
from pyspark.sql import functions as F

# Import project-specific modules
from src.utils import create_spark_session, setup_logging
from config.app_config import PIPELINE_PATH, ENSEMBLE_MODEL_PATHS,CHART_COLORS, Conda_env_path

os.environ['PYSPARK_PYTHON'] = Conda_env_path
os.environ['PYSPARK_DRIVER_PYTHON'] = Conda_env_path

# Initialize logging
logger = setup_logging()

# Initialize Flask app
app = Flask(__name__)

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

    def predict(self, lyrics_text: str) -> dict:
        """
        Makes a final prediction using the voting ensemble.
        Returns a dictionary with the prediction and probabilities.
        """
        # Create a DataFrame for the new lyrics
        new_song_df = spark.createDataFrame([(lyrics_text,)], ["lyrics"])
        
        # 1. Apply the feature engineering pipeline
        self.logger.info("Applying feature transformations...")
        transformed_data = self.pipeline.transform(new_song_df)

        # 2. Get predictions from each base model
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
        
        # 5. Convert prediction index back to genre string
        string_indexer_model = self.pipeline.stages[0]
        label_converter = IndexToString(
            inputCol="prediction",
            outputCol="genre_prediction",
            labels=string_indexer_model.labels
        )
        final_prediction_df = label_converter.transform(final_preds_indexed)
        
        # Convert individual predictions to genre strings
        gbt_converter = IndexToString(
            inputCol="gbt_pred",
            outputCol="gbt_genre",
            labels=string_indexer_model.labels
        )
        lr_converter = IndexToString(
            inputCol="lr_pred",
            outputCol="lr_genre",
            labels=string_indexer_model.labels
        )
        nb_converter = IndexToString(
            inputCol="nb_pred",
            outputCol="nb_genre",
            labels=string_indexer_model.labels
        )
        
        final_prediction_df = gbt_converter.transform(final_prediction_df)
        final_prediction_df = lr_converter.transform(final_prediction_df)
        final_prediction_df = nb_converter.transform(final_prediction_df)
        
        # Extract all predictions
        row = final_prediction_df.first()
        predicted_genre = row["genre_prediction"]
        gbt_genre = row["gbt_genre"]
        lr_genre = row["lr_genre"]
        nb_genre = row["nb_genre"]
        
        
        return {
            "ensemble_prediction": predicted_genre,
            "individual_predictions": {
                "gradient_boosted_trees": gbt_genre,
                "logistic_regression": lr_genre,
                "naive_bayes": nb_genre
            },
        }

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', chart_colors=CHART_COLORS)

@app.route('/classify', methods=['POST'])
def classify_lyrics():
    """API endpoint for classifying lyrics."""
    try:
        data = request.get_json()
        lyrics = data.get('lyrics', '')
        
        if not lyrics or len(lyrics.strip()) < 10:
            return jsonify({'error': 'Please provide at least 10 characters of lyrics.'}), 400
        
        logging.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = predictor.predict(lyrics)
        logging.info(f"ensemble prediction: {result['ensemble_prediction']}")
        logging.info(f"individual predictions: {result['individual_predictions']}")
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error during classification: {str(e)}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during classification.'}), 500

if __name__ == '__main__':
    try:
        setup_logging()

        logging.info("Initializing Spark session for the Flask app...")
        spark = create_spark_session()
        
        logging.info("Loading models into memory...")
        predictor = EnsemblePredictor()
        
        logging.info("Starting Flask web server on http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)

    except KeyboardInterrupt:
        logging.info("Shutting down web server")

    finally:
        if 'spark' in locals():
            spark.stop()
            logging.info("Spark session stopped")