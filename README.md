# Emotion-Analysis
Analysed Human emotion using machine learning concepts

🤖 Real-Time Emotion Detection and Time-Series Analytics
🌟 Project Goal
This project implements a two-phase system for emotional intelligence:

Real-Time Detection: Use Computer Vision (OpenCV) and Deep Learning (Keras/TensorFlow) to detect faces in a video stream and classify their emotion.

Advanced Analytics: Perform time-series analysis and Exploratory Data Analysis (EDA) on the collected emotion data to identify patterns, behavioral trends, and make future forecasts.

This is ideal for analyzing customer behavior, audience engagement, or general mood over time.

🚀 Getting Started
Prerequisites
You'll need Python 3.8 - 3.11 and the following core libraries.

Package

Purpose

opencv-python

Video processing and real-time visualization.

tensorflow / keras

Loading the pre-trained emotion model (.h5).

pandas / seaborn

Data loading, cleaning, and sophisticated visualization.

statsmodels

Time series decomposition and forecasting (ARIMA).

Install everything using pip:

pip install opencv-python tensorflow pandas numpy matplotlib seaborn statsmodels

Setup and Required Assets
For the detection script (facedetection.ipynb) to run, you must download the pre-trained neural network assets and place them in a dedicated folder:

Create a folder named models/ in the root directory.

Download the face detection model files:

models/deploy.prototxt

models/res10_300x300_ssd_iter_140000.caffemodel

Download the emotion classification model:

models/emotion_model.h5

🛠️ Project Structure & Execution
The project is structured into two main Jupyter Notebooks:

1. facedetection.ipynb (Real-Time CV)
This notebook handles the live processing pipeline:

Face Detection: Uses a Caffe-based Single Shot Detector (SSD) model (loaded from .prototxt and .caffemodel) to locate faces efficiently within each frame.

Emotion Classification: Extracts the detected face, converts it to a 48x48 pixel grayscale image, normalizes the pixels, and feeds it into the Keras model (emotion_model.h5) to predict one of the seven primary emotions (Angry, Happy, Neutral, Sad, etc.).

Output: Overlays the detected bounding box and the predicted emotion label directly onto the video stream.

2. emotionAnalysis.ipynb (Data Science)
This notebook is where the collected data (emotion_dataset.csv) is converted into actionable intelligence:

Data Preparation: Converts timestamps to date/hour features.

Key Visualizations:

Emotion Distribution: Checks the overall frequency of each emotion (notably showing a high "neutral" bias).

Time-of-Day Trends: Plots emotion counts by hour and day of the week to reveal when specific emotions are most frequent.

Time Series Deep Dive (The cool part!):

Seasonal Decomposition: Separates the emotion data (e.g., 'neutral' counts) into Trend, Seasonality (7-day cycle), and Residual components to understand underlying forces.

Forecasting: Fits an ARIMA model to the data to predict emotion counts for the next 7 days.

📊 Analytics Highlights
The EDA confirms that human subjects are predominantly registered as Neutral, but clear, actionable patterns emerge:

Weekly Seasonality: There is a distinct 7-day cycle in the emotional data, suggesting that mood is influenced by the day of the week (e.g., weekends vs. weekdays).

Predictive Power: The ARIMA forecast demonstrates the potential to predict short-term emotional baselines, which is valuable for planning staffing or content deployment.

Customer Profiling: The script tracks emotions per customer_id, allowing for identification of the most emotionally expressive or "neutral" individuals in the dataset.
