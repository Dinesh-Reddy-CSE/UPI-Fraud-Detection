import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Extract time-based features
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
    df_processed['is_night'] = ((df_processed['hour'] >= 0) & (df_processed['hour'] <= 5)).astype(int)
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['merchant_category', 'device', 'location']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Calculate user transaction frequency
    user_freq = df_processed['user_id'].value_counts().to_dict()
    df_processed['user_frequency'] = df_processed['user_id'].map(user_freq)
    
    # Feature set for model training
    features = ['amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
                'merchant_category', 'device', 'location', 'user_frequency']
    
    X = df_processed[features]
    
    return X, label_encoders

def train_isolation_forest(df, contamination=0.05):
    # Prepare features
    X, label_encoders = prepare_features(df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest model
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42
    )
    
    model.fit(X_scaled)
    
    # Predict anomalies (1 for normal, -1 for anomaly)
    predictions = model.predict(X_scaled)
    
    # Convert to binary (0 for normal, 1 for fraud)
    df['predicted_fraud'] = [1 if x == -1 else 0 for x in predictions]
    
    # Evaluate model if we have ground truth
    if 'is_fraud' in df.columns:
        print("Model Evaluation:")
        print(classification_report(df['is_fraud'], df['predicted_fraud']))
        print("Confusion Matrix:")
        print(confusion_matrix(df['is_fraud'], df['predicted_fraud']))
    
    # Save model and encoders
    joblib.dump(model, 'isolation_forest_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return model, label_encoders

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('upi_transactions.csv')
    print(f"Data loaded with {len(df)} records")
    
    print("Training Isolation Forest model...")
    model, label_encoders = train_isolation_forest(df)
    print("Model training completed and saved!")