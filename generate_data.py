import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_upi_data(num_records=10000):
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate user IDs
    user_ids = [f'USER{str(i).zfill(5)}' for i in range(1, 501)]
    
    # Merchant categories
    merchant_categories = ['Groceries', 'Electronics', 'Food', 'Transport', 
                          'Healthcare', 'Entertainment', 'Utilities', 'Retail']
    
    # Device types
    devices = ['Android', 'iOS', 'Web']
    
    # Indian cities
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
             'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kochi', 'Bhopal']
    
    # Generate timestamps over the last 90 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    
    timestamps = []
    for _ in range(num_records):
        random_seconds = random.randint(0, 90*24*60*60)
        timestamp = start_time + timedelta(seconds=random_seconds)
        timestamps.append(timestamp)
    
    # Generate transaction amounts (most transactions are small)
    amounts = np.concatenate([
        np.random.exponential(500, int(num_records * 0.8)),  # 80% small transactions
        np.random.uniform(5000, 100000, int(num_records * 0.2))  # 20% larger transactions
    ])
    np.random.shuffle(amounts)
    
    # Create base dataframe
    data = {
        'transaction_id': [f'TXN{str(i).zfill(7)}' for i in range(1, num_records+1)],
        'timestamp': timestamps,
        'amount': amounts,
        'user_id': np.random.choice(user_ids, num_records),
        'merchant_category': np.random.choice(merchant_categories, num_records),
        'device': np.random.choice(devices, num_records),
        'location': np.random.choice(cities, num_records),
        'ip_address': [f'{random.randint(100, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}' 
                      for _ in range(num_records)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some patterns for normal behavior
    # Most transactions happen during day time (9 AM to 9 PM)
    df['hour'] = df['timestamp'].dt.hour
    df['is_daytime'] = df['hour'].between(9, 21)
    
    # Add some frequency features
    user_freq = df['user_id'].value_counts().to_dict()
    df['user_transaction_count'] = df['user_id'].map(user_freq)
    
    # Create some fraudulent patterns
    fraud_indices = []
    
    # 1. Unusual time transactions (between 1 AM and 5 AM)
    night_indices = df[df['hour'].between(1, 5)].sample(frac=0.05).index
    fraud_indices.extend(night_indices)
    
    # 2. Large amount transactions
    large_amount_indices = df[df['amount'] > 50000].sample(frac=0.1).index
    fraud_indices.extend(large_amount_indices)
    
    # 3. New device transactions (simulate by making device uncommon for user)
    user_device_combinations = df.groupby(['user_id', 'device']).size().reset_index()
    common_devices = user_device_combinations.groupby('user_id')['device'].first().to_dict()
    
    new_device_indices = []
    for idx, row in df.iterrows():
        if common_devices.get(row['user_id']) and row['device'] != common_devices[row['user_id']]:
            if random.random() < 0.05:  # 5% chance of fraud when using new device
                new_device_indices.append(idx)
    
    fraud_indices.extend(new_device_indices)
    
    # 4. Unusual location transactions
    user_city_combinations = df.groupby(['user_id', 'location']).size().reset_index()
    common_cities = user_city_combinations.groupby('user_id')['location'].first().to_dict()
    
    new_city_indices = []
    for idx, row in df.iterrows():
        if common_cities.get(row['user_id']) and row['location'] != common_cities[row['user_id']]:
            if random.random() < 0.03:  # 3% chance of fraud when in new city
                new_city_indices.append(idx)
    
    fraud_indices.extend(new_city_indices)
    
    # Remove duplicates
    fraud_indices = list(set(fraud_indices))
    
    # Mark fraud transactions
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Ensure we have about 5% fraud rate
    fraud_rate = df['is_fraud'].mean()
    if fraud_rate < 0.05:
        # Add more fraud cases if needed
        needed_frauds = int(0.05 * num_records) - len(fraud_indices)
        non_fraud_indices = df[df['is_fraud'] == 0].index
        additional_frauds = np.random.choice(non_fraud_indices, needed_frauds, replace=False)
        df.loc[additional_frauds, 'is_fraud'] = 1
    
    # Drop temporary columns
    df = df.drop(['hour', 'is_daytime'], axis=1)
    
    return df

if __name__ == "__main__":
    print("Generating UPI transaction data...")
    df = generate_upi_data(20000)
    df.to_csv('upi_transactions.csv', index=False)
    print(f"Data generated with {len(df)} records")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")