# 🎵 Music Genre Classifier (PySpark + Flask)

A complete ML web app that predicts a song’s **genre from its lyrics**.
Built with **PySpark MLlib** for data processing & training and **Flask** for serving an **ensemble** of three models: **GBTClassifier**, **Logistic Regression**, and **Naive Bayes**.
The app is **Dockerized** for one-command setup.

---

## ✨ Key Features
- **Big-Data Ready:** PySpark pipelines that scale past RAM limits
- **Ensemble Modeling:** Majority voting across GBT, LR, and NB
- **Interactive Web App:** Paste lyrics, get predictions + model votes
- **Dockerized:** Consistent, reproducible environment
- **Modular Codebase:** Clear `src/` scripts for processing, features, training, and prediction

---

## 🧱 Tech Stack
- **Backend:** Python, Flask
- **ML:** Apache Spark (PySpark MLlib)
- **Container:** Docker
- **Frontend:** HTML, CSS, JavaScript

---

## 📸 Screenshots

![Web App Screenshot 1](images/screenshot_1.png)
![Web App Screenshot 2](images/screenshot_2.png)

---

## 📦 Project Structure

```text
.
├── app.py                  # Main Flask application logic
├── Dockerfile 
├── notebooks               # jupyter notebooks 
├── requirements.txt        # Python package dependencies
├── config/                 # Configuration file
│   └── app_config.py
├── data/                   # Directory for datasets
├── models/                 # Saved machine learning models
├── src/                    # ML pipeline scripts
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── training.py
│   └── predict.py
├── static/                 # Frontend assets
│   ├── css/style.css
│   └── js/scripts.js
│  
└── templates/              # HTML files for the 
    └── index.html
```
---

##  Getting Started

You can run the app with **Docker** or **locally**.

### Option 1 — Run with Docker 🐳
1.  **Build the image:**
    ```bash
    docker build -t music-classifier .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 5000:5000 music-classifier
    ```
3.  **Open in browser:**
    `http://localhost:5000`

### Option 2 — Local Setup ⚙️
1.  **Clone the repo**

2.  **Create venv + install dependencies**


#### 🔄 Local ML Workflow
1.  **Add Data** → place your dataset into the `data/` directory.
2.  **Process Data**
    ```bash
    python src/data_processing.py
    ```
3.  **Build Features**
    ```bash
    python src/feature_engineering.py
    ```
4.  **Train Models**
    ```bash
    python src/training.py
    ```
5.  **Run Web App**
    ```bash
    python app.py
    ```
    Then open in your browser: `http://localhost:5000`