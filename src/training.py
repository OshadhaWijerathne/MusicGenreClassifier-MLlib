import os
import logging
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import create_spark_session, setup_logging

class ModelTrainer:
    """Handles the model training and evaluation process."""

    def __init__(self, spark: SparkSession, pipeline_path: str, train_path: str, test_path: str):
        self.spark = spark
        self.pipeline_path = pipeline_path
        self.train_path = train_path
        self.test_path = test_path
        self.logger = logging.getLogger(__name__)

    def _load_and_transform_data(self):
        """Loads data and the feature pipeline, then transforms the data."""
        self.logger.info(f"Loading feature pipeline from {self.pipeline_path}")
        pipeline_model = PipelineModel.load(self.pipeline_path)

        self.logger.info(f"Loading training data from {self.train_path}")
        df_train = self.spark.read.csv(self.train_path, header=True, inferSchema=True)

        self.logger.info(f"Loading test data from {self.test_path}")
        df_test = self.spark.read.csv(self.test_path, header=True, inferSchema=True)

        self.logger.info("Applying transformations to train and test data.")
        self.train_transformed = pipeline_model.transform(df_train)
        self.test_transformed = pipeline_model.transform(df_test)

    def train_and_evaluate(self):
        """Trains and evaluates a suite of classification models."""
        self._load_and_transform_data()
        
        models = {
            "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", seed=42),
            "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label"),
            "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="label"),
        }
    
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        for name, model in models.items():
            self.logger.info("-" * 30)
            self.logger.info(f"Training {name}...")
            
            fitted_model = model.fit(self.train_transformed)
            predictions = fitted_model.transform(self.test_transformed)
            f1_score = evaluator.evaluate(predictions)
            
            self.logger.info(f" F1-Score for {name}: {f1_score:.4f}")
        
        self.logger.info("-" * 30)

    def run(self):
        """Executes the full model training workflow."""
        self.logger.info("--- Starting Model Training Workflow ---")
        self.train_and_evaluate()
        self.logger.info("--- Model Training Workflow Complete ---")


if __name__ == "__main__":
    setup_logging()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_train.csv")
    TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_test.csv")
    PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_pipeline_lyrics_only")

    spark_session = create_spark_session("MusicClassifier_Training")

    trainer = ModelTrainer(spark_session, PIPELINE_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH)
    trainer.run()
    
    spark_session.stop()