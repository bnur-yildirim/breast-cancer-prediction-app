# Breast Cancer Wisconsin Diagnosis – Machine Learning and Deep Learning Web App

This project is an interactive web application built with Streamlit that demonstrates the use of machine learning and deep learning models for breast cancer classification. It is based on the Breast Cancer Wisconsin Diagnostic Dataset. The app allows users to visualize the dataset, compare trained models, and make predictions either by uploading a CSV file or manually entering feature values.

## Project Features

- Interactive and user-friendly Streamlit web app  
- Data visualization including class distribution and correlation heatmap  
- Comparison of trained models based on various performance metrics  
- Selection of best or worst model based on chosen metric  
- Prediction on new data using selected model or all models  
- Option to enter feature values manually or upload a CSV file for prediction  
- Handles missing values using predefined mean values from the dataset  

## Setup Instructions

### Step 1: Download or Clone the Project

You can either download the entire project as a ZIP file and extract it, or clone it from your repository:

```bash
git clone https://github.com/your-username/breast-cancer-app.git
cd breast-cancer-app

## Step 2: Create and Activate the Conda Environment

Make sure you have Anaconda or Miniconda installed.

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate baykar

## How to Run the Application

```bash
streamlit run app/app.py

## Dataset Information

The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset, which contains 30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses. The goal is to classify tumors as malignant or benign.

### Dataset source:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)