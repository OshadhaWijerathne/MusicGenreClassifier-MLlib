# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Java (needed for Spark)
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*

# Set environment variables for Java + Spark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config/ /app/config/
COPY models/ /app/models/
COPY src/utils.py /src/utils.py
COPY static/ /app/static/
COPY templates/ /app/templates/
COPY app.py requirements.txt /app/


# Expose port
EXPOSE 5000

# Start Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
