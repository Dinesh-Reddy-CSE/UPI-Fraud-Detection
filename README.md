# UPI Fraud Detection System 🛡️💳

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A Machine Learning-based system designed to detect fraudulent UPI (Unified Payments Interface) transactions in real-time. Built to enhance digital payment security by identifying suspicious patterns using supervised learning models.

---

## 📌 Overview

With the rapid adoption of UPI in India, fraud detection has become critical for financial institutions and users alike. This project leverages historical transaction data to train classification models that can flag potentially fraudulent UPI transactions before they cause harm.

Built using Python, Scikit-learn, Pandas, and deployed with a Flask API for easy integration.

---

## ✅ Features

- Real-time UPI transaction fraud detection
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Feature engineering for transaction behavior analysis
- Model evaluation with precision, recall, and F1-score
- REST API for integration with banking/payment apps
- Configurable thresholds for fraud probability

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Data Processing**: Pandas, NumPy  
- **Machine Learning**: Scikit-learn  
- **Model Serialization**: Joblib  
- **Visualization**: Matplotlib, Seaborn  
- **Web Framework**: Streamlit  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dinesh-Reddy-CSE/UPI-Fraud-Detection.git
   cd UPI-Fraud-Detection

2. Install requirements:
   ```bash
   pip install -r requirements.txt

3. Train the model:
    ```bash
    python train_model.py

4. Start the API Server:
    ```bash
    python app.py

5. Access the API at:
   ```bash
    [python app.py](http://localhost:5000/predict)

