import os

# --- Conda ENV ---
Conda_env_path = "C:/Users/rwkos/miniconda3/envs/music_classifier/python.exe"

# --- Project Root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Data File Paths ---
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_dataset.csv")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_train.csv")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Mendeley_cleaned_test.csv")

# --- Model and Pipeline Paths ---
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_pipeline_lyrics_only")
#SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model")

# --- Paths for each model in the Ensemble ---
ENSEMBLE_MODEL_PATHS = {
    "gbt": os.path.join(PROJECT_ROOT, "models", "ensemble_gbt"),
    "lr": os.path.join(PROJECT_ROOT, "models", "ensemble_lr"),
    "nb": os.path.join(PROJECT_ROOT, "models", "ensemble_nb")
}

# --- Spark Configuration ---
SPARK_CONFIG = {
    "app_name": "MusicGenreClassifier",
    "master": "local[2]",
    "driver_memory": "4g"
}

# --- Feature Engineering Parameters ---
FEATURE_CONFIG = {
    "hashing_tf_features": 10000,
    "random_split_seed": 42
}

# --- Model Training Parameters ---
MODEL_CONFIG = {
    "random_forest_seed": 42,
    "gbt_seed": 42
}

# --- Web App Settings ---
# Chart colors for visualization
CHART_COLORS = {
    'pop': '#FF6384',
    'country': '#36A2EB',
    'blues': '#FFCE56',
    'jazz': '#4BC0C0',
    'reggae': '#9966FF',
    'rock': '#FF9F40',
    'hip hop': '#C9CBCF'
}