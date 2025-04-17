import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, 
    NaiveBayes,
    GBTClassifier,      
    OneVsRest           
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import utility functions and config variables
from src.utils import create_spark_session, setup_logging
from config.app_config import (
    TRAIN_DATA_PATH, 
    TEST_DATA_PATH, 
    PIPELINE_PATH, 
    MODEL_CONFIG,
    ENSEMBLE_MODEL_PATHS,Conda_env_path
)
from pyspark.sql import functions as F

import os
import sys
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = Conda_env_path
os.environ['PYSPARK_DRIVER_PYTHON'] = Conda_env_path

class ModelTrainer:
    """Handles model training with model ensemble."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self._load_and_transform_data()

    def _load_and_transform_data(self):
        """Loads and transforms data using the saved pipeline."""
        self.logger.info(f"Loading feature pipeline from {PIPELINE_PATH}")
        pipeline_model = PipelineModel.load(PIPELINE_PATH)

        self.logger.info(f"Loading training data from {TRAIN_DATA_PATH}")
        df_train = self.spark.read.csv(TRAIN_DATA_PATH, header=True, inferSchema=True)

        self.logger.info(f"Loading test data from {TEST_DATA_PATH}")
        df_test = self.spark.read.csv(TEST_DATA_PATH, header=True, inferSchema=True)

        self.logger.info("Applying transformations to train and test data.")
        self.train_transformed = pipeline_model.transform(df_train).cache()
        self.test_transformed = pipeline_model.transform(df_test).cache()

    def train_and_evaluate_individuals(self):
        """Trains and evaluates a suite of individual classification models."""
        self.logger.info("--- Training and Evaluating Individual Models ---")
        
        gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=MODEL_CONFIG["gbt_seed"])
        ovr_gbt = OneVsRest(classifier=gbt)
        
        models = {
            "GBT with One-vs-Rest": ovr_gbt, 
            "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label"),
            "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="label"),
        }
    
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            fitted_model = model.fit(self.train_transformed)
            predictions = fitted_model.transform(self.test_transformed)
            f1_score = evaluator.evaluate(predictions)
            self.logger.info(f" F1-Score for {name}: {f1_score:.4f}")
        
        self.logger.info("-" * 30)

    def train_and_evaluate_ensemble(self):
        """Trains an ensemble model using majority voting."""
        self.logger.info("--- Training and Evaluating Voting Ensemble ---")

        # --- CHANGE: Define base models for the new ensemble ---
        ovr_gbt_model = OneVsRest(classifier=GBTClassifier(featuresCol="features", labelCol="label", seed=MODEL_CONFIG["gbt_seed"]))
        lr_model = LogisticRegression(featuresCol="features", labelCol="label")
        nb_model = NaiveBayes(featuresCol="features", labelCol="label")

        self.logger.info("Training base models for the ensemble...")
        fitted_ovr_gbt = ovr_gbt_model.fit(self.train_transformed)
        fitted_lr = lr_model.fit(self.train_transformed)
        fitted_nb = nb_model.fit(self.train_transformed)

        # --- Save each fitted model ---
        self.logger.info(f"Saving OvR-GBT model to {ENSEMBLE_MODEL_PATHS['gbt']}")
        fitted_ovr_gbt.write().overwrite().save(ENSEMBLE_MODEL_PATHS['gbt'])

        self.logger.info(f"Saving Logistic Regression model to {ENSEMBLE_MODEL_PATHS['lr']}")
        fitted_lr.write().overwrite().save(ENSEMBLE_MODEL_PATHS['lr'])
        
        self.logger.info(f"Saving Naive Bayes model to {ENSEMBLE_MODEL_PATHS['nb']}")
        fitted_nb.write().overwrite().save(ENSEMBLE_MODEL_PATHS['nb'])

        self.logger.info("Generating predictions from each model...")
        preds_ovr_gbt = fitted_ovr_gbt.transform(self.test_transformed).withColumnRenamed("prediction", "gbt_pred")
        preds_lr = fitted_lr.transform(self.test_transformed).withColumnRenamed("prediction", "lr_pred")
        preds_nb = fitted_nb.transform(self.test_transformed).withColumnRenamed("prediction", "nb_pred")
        
        # Add a unique ID to each DataFrame to ensure a correct row-by-row join
        preds_ovr_gbt = preds_ovr_gbt.withColumn("id", F.monotonically_increasing_id())
        preds_lr = preds_lr.withColumn("id", F.monotonically_increasing_id())
        preds_nb = preds_nb.withColumn("id", F.monotonically_increasing_id())

        combined_preds = preds_ovr_gbt.select("id", "label", "gbt_pred") \
            .join(preds_lr.select("id", "lr_pred"), "id") \
            .join(preds_nb.select("id", "nb_pred"), "id")

        # Define the voting UDF (User-Defined Function)
        def majority_vote(*predictions):
            from collections import Counter
            return Counter(predictions).most_common(1)[0][0]

        vote_udf = udf(majority_vote, DoubleType())

        self.logger.info("Applying majority vote to combined predictions...")
        final_preds = combined_preds.withColumn(
            "prediction", 
            vote_udf(col("gbt_pred"), col("lr_pred"), col("nb_pred")) 
        )

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        ensemble_f1 = evaluator.evaluate(final_preds)
        
        self.logger.info(f'F1-Score for Voting Ensemble (OvR-GBT+LR+NB): {ensemble_f1:.4f}')

    def run(self):
        """Executes the full model training and evaluation workflow."""
        #self.train_and_evaluate_individuals()
        self.train_and_evaluate_ensemble()


if __name__ == "__main__":
    setup_logging()
    
    spark_session = create_spark_session()
    trainer = ModelTrainer(spark_session)
    trainer.run()
    spark_session.stop()