import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time

# Set page config
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('upi_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load('isolation_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

# Preprocess new transaction for prediction
def preprocess_transaction(transaction, label_encoders):
    # Create a copy
    transaction_df = transaction.copy()
    
    # Extract time-based features
    transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])
    transaction_df['hour'] = transaction_df['timestamp'].dt.hour
    transaction_df['day_of_week'] = transaction_df['timestamp'].dt.dayofweek
    transaction_df['is_weekend'] = transaction_df['day_of_week'].isin([5, 6]).astype(int)
    transaction_df['is_night'] = ((transaction_df['hour'] >= 0) & (transaction_df['hour'] <= 5)).astype(int)
    
    # Encode categorical features using pre-trained encoders
    categorical_cols = ['merchant_category', 'device', 'location']
    
    for col in categorical_cols:
        le = label_encoders[col]
        # Handle unseen labels by using a default value
        try:
            transaction_df[col] = le.transform([transaction_df[col]])[0]
        except ValueError:
            # If label is unseen, use the most frequent category
            transaction_df[col] = le.transform([le.classes_[0]])[0]
    
    # Calculate user transaction frequency (this would normally come from historical data)
    # For demo purposes, we'll use a placeholder value
    transaction_df['user_frequency'] = 10  # Placeholder
    
    # Feature set for prediction
    features = ['amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
                'merchant_category', 'device', 'location', 'user_frequency']
    
    return transaction_df[features]

def main():
    st.title("UPI Transaction Fraud Detection System")
    st.markdown("""
    This system detects anomalous UPI transactions that may indicate fraudulent activity.
    It uses Isolation Forest, an unsupervised machine learning algorithm, to identify unusual patterns.
    """)
    
    # Load data and model
    df = load_data()
    model, label_encoders = load_model()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Dashboard", "Transaction Analysis", "Real-time Detection"])
    
    if app_mode == "Dashboard":
        show_dashboard(df)
    elif app_mode == "Transaction Analysis":
        show_transaction_analysis(df)
    elif app_mode == "Real-time Detection":
        show_real_time_detection(df, model, label_encoders)

def show_dashboard(df):
    st.header("Transaction Dashboard")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df)
    fraud_transactions = df['is_fraud'].sum()
    fraud_rate = (fraud_transactions / total_transactions) * 100
    
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraudulent Transactions", f"{fraud_transactions:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Transaction Amount", f"₹{df['amount'].mean():.2f}")
    
    # Time series of transactions
    st.subheader("Transaction Trends Over Time")
    
    df_daily = df.set_index('timestamp').resample('D').agg({
        'transaction_id': 'count',
        'is_fraud': 'sum',
        'amount': 'mean'
    }).rename(columns={'transaction_id': 'count', 'amount': 'avg_amount'})
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df_daily.index, y=df_daily['count'], name="Transaction Count"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df_daily.index, y=df_daily['is_fraud'], name="Fraud Count"),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Daily Transaction Volume and Fraud Count"
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Count", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fraud by category
    st.subheader("Fraud Distribution by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_by_category = df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)
        fig = px.bar(fraud_by_category, 
                     title="Fraud Rate by Merchant Category",
                     labels={'value': 'Fraud Rate', 'merchant_category': 'Category'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fraud_by_device = df.groupby('device')['is_fraud'].mean().sort_values(ascending=False)
        fig = px.pie(fraud_by_device, 
                     names=fraud_by_device.index, 
                     values=fraud_by_device.values,
                     title="Fraud Distribution by Device")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud heatmap by hour and day
    st.subheader("Fraud Heatmap by Hour and Day of Week")
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    fraud_heatmap_data = df.groupby(['day_of_week', 'hour'])['is_fraud'].mean().unstack()
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fraud_heatmap_data.index = days
    
    fig = px.imshow(fraud_heatmap_data, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Fraud Rate"),
                    title="Fraud Rate by Day and Hour")
    
    st.plotly_chart(fig, use_container_width=True)

def show_transaction_analysis(df):
    st.header("Transaction Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    
    with col2:
        min_amount, max_amount = st.slider(
            "Transaction Amount Range (₹)",
            min_value=0,
            max_value=int(df['amount'].max()),
            value=(0, int(df['amount'].max()))
        )
    
    with col3:
        fraud_only = st.checkbox("Show only fraudulent transactions")
    
    # Apply filters
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) & 
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    filtered_df = filtered_df[
        (filtered_df['amount'] >= min_amount) & 
        (filtered_df['amount'] <= max_amount)
    ]
    
    if fraud_only:
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
    
    # Show filtered data
    st.subheader(f"Filtered Transactions ({len(filtered_df)} records)")
    st.dataframe(filtered_df.head(100))
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )
    
    # Fraud patterns analysis
    st.subheader("Fraud Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top fraudulent users
        fraudulent_users = filtered_df[filtered_df['is_fraud'] == 1]['user_id'].value_counts().head(10)
        fig = px.bar(fraudulent_users, 
                     title="Top Users with Fraudulent Transactions",
                     labels={'value': 'Fraud Count', 'user_id': 'User ID'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud by location
        fraud_by_location = filtered_df[filtered_df['is_fraud'] == 1]['location'].value_counts().head(10)
        fig = px.pie(fraud_by_location, 
                     names=fraud_by_location.index, 
                     values=fraud_by_location.values,
                     title="Fraud Distribution by Location")
        st.plotly_chart(fig, use_container_width=True)

def show_real_time_detection(df, model, label_encoders):
    st.header("Real-time Fraud Detection")
    
    st.markdown("""
    Simulate a new transaction to check if it would be flagged as fraudulent.
    The system uses a pre-trained Isolation Forest model to detect anomalies.
    """)
    
    # Create a form for transaction details
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.selectbox("User ID", df['user_id'].unique())
            amount = st.number_input("Amount (₹)", min_value=1.0, max_value=1000000.0, value=1000.0)
            merchant_category = st.selectbox("Merchant Category", df['merchant_category'].unique())
        
        with col2:
            device = st.selectbox("Device", df['device'].unique())
            location = st.selectbox("Location", df['location'].unique())
            transaction_date = st.date_input("Transaction Date", datetime.now().date())
            transaction_time = st.time_input("Transaction Time", datetime.now().time())
        
        # Add submit button
        submitted = st.form_submit_button("Check for Fraud")
    
    if submitted:
        # Combine date and time
        transaction_datetime = datetime.combine(transaction_date, transaction_time)
        
        # Create transaction object
        transaction = {
            'timestamp': transaction_datetime,
            'amount': amount,
            'user_id': user_id,
            'merchant_category': merchant_category,
            'device': device,
            'location': location
        }
        
        # Preprocess transaction
        transaction_df = pd.DataFrame([transaction])
        features = preprocess_transaction(transaction_df, label_encoders)
        
        # Predict
        prediction = model.predict(features)
        is_fraud = prediction[0] == -1
        
        # Display result
        if is_fraud:
            st.error("🚨 Fraud Alert: This transaction has been flagged as potentially fraudulent!")
            
            # Explain why
            st.subheader("Why was this transaction flagged?")
            
            reasons = []
            
            # Check for unusual time
            hour = transaction_datetime.hour
            if hour < 6 or hour > 22:
                reasons.append(f"Unusual transaction time: {hour}:00")
            
            # Check for large amount
            user_avg_amount = df[df['user_id'] == user_id]['amount'].mean()
            if amount > user_avg_amount * 3:
                reasons.append(f"Large amount compared to user's average: ₹{user_avg_amount:.2f}")
            
            # Check for new device
            user_devices = df[df['user_id'] == user_id]['device'].unique()
            if device not in user_devices:
                reasons.append(f"New device not previously used by this user")
            
            # Check for new location
            user_locations = df[df['user_id'] == user_id]['location'].unique()
            if location not in user_locations:
                reasons.append(f"New location not previously used by this user")
            
            if reasons:
                for reason in reasons:
                    st.write(f"- {reason}")
            else:
                st.write("- Multiple unusual patterns detected")
        
        else:
            st.success("✅ Transaction appears legitimate")
            
            # Show confidence factors
            st.subheader("Why was this transaction approved?")
            
            approvals = []
            
            # Check for normal time
            hour = transaction_datetime.hour
            if 9 <= hour <= 18:
                approvals.append(f"Normal transaction time: {hour}:00")
            
            # Check for normal amount
            user_avg_amount = df[df['user_id'] == user_id]['amount'].mean()
            if amount <= user_avg_amount * 1.5:
                approvals.append(f"Amount consistent with user's spending pattern: ₹{user_avg_amount:.2f} average")
            
            # Check for known device
            user_devices = df[df['user_id'] == user_id]['device'].unique()
            if device in user_devices:
                approvals.append(f"Device previously used by this user")
            
            # Check for known location
            user_locations = df[df['user_id'] == user_id]['location'].unique()
            if location in user_locations:
                approvals.append(f"Location previously used by this user")
            
            for approval in approvals:
                st.write(f"- {approval}")

if __name__ == "__main__":
    main()