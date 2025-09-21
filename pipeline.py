import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['timestamp'] = pd.to_datetime(X_['timestamp'])
        X_['hour'] = X_['timestamp'].dt.hour
        X_['day_of_week'] = X_['timestamp'].dt.dayofweek
        X_['is_weekend'] = X_['day_of_week'].isin([5, 6]).astype(int)
        X_['is_night'] = ((X_['hour'] >= 0) & (X_['hour'] <= 5)).astype(int)
        return X_.drop(columns=['timestamp'])

class UserFrequencyCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.user_frequency_ = None

    def fit(self, X, y=None):
        self.user_frequency_ = X['user_id'].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['user_frequency'] = X_['user_id'].map(self.user_frequency_).fillna(0)
        return X_

# Define the columns for each transformation
numeric_features = ['amount', 'hour', 'day_of_week', 'user_frequency']
categorical_features = ['merchant_category', 'device', 'location']

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ],
    remainder='passthrough'
)

def create_pipeline():
    return Pipeline(steps=[
        ('user_frequency', UserFrequencyCalculator()),
        ('time_extractor', TimeFeatureExtractor()),
        ('column_dropper', ColumnTransformer([('drop_columns', 'drop', ['user_id'])], remainder='passthrough')),
        ('preprocessor', preprocessor),
        ('model', IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, random_state=42))
    ])
