import pandas as pd
import joblib
from pipeline import create_pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(df):
    # Create the pipeline
    pipeline = create_pipeline()
    
    # Define features
    X = df.drop(columns=['is_fraud', 'transaction_id'], errors='ignore')
    
    # Fit the pipeline
    pipeline.fit(X)
    
    # Predict anomalies
    # Note: In IsolationForest, 1 is inlier, -1 is outlier. We map -1 to 1 (fraud) and 1 to 0 (not fraud).
    predictions = pipeline.predict(X)
    df['predicted_fraud'] = [1 if x == -1 else 0 for x in predictions]
    
    # Evaluate model if 'is_fraud' column exists
    if 'is_fraud' in df.columns:
        print("Model Evaluation:")
        print(classification_report(df['is_fraud'], df['predicted_fraud']))
        print("Confusion Matrix:")
        print(confusion_matrix(df['is_fraud'], df['predicted_fraud']))
    
    # Save the entire pipeline
    joblib.dump(pipeline, 'fraud_detection_pipeline.pkl')
    
    return pipeline

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('upi_transactions.csv')
    print(f"Data loaded with {len(df)} records")
    
    print("Training the fraud detection pipeline...")
    train_and_evaluate(df)
    print("Pipeline training completed and saved as fraud_detection_pipeline.pkl!")