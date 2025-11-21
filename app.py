import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

st.set_page_config(page_title="NYC Taxi Trip Duration Predictor (Ridge)", layout="wide")

st.title("🚕 NYC Taxi Trip Duration Predictor (Ridge Regression)")

MODEL_PATH = 'models/approach1_model_ridge_20000_fixed.pkl'

@st.cache_resource
def load_model(model_path=MODEL_PATH):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def prepare_input(pickup_datetime, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, passenger_count):
    input_data = pd.DataFrame({
        'pickup_datetime': [pickup_datetime],
        'pickup_latitude': [pickup_latitude],
        'pickup_longitude': [pickup_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'passenger_count': [passenger_count]
    })

    input_data['pickup_datetime'] = pd.to_datetime(input_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    input_data['dayofweek'] = input_data.pickup_datetime.dt.dayofweek
    input_data['month'] = input_data.pickup_datetime.dt.month
    input_data['hour'] = input_data.pickup_datetime.dt.hour
    input_data['dayofyear'] = input_data.pickup_datetime.dt.dayofyear

    lat1, lon1 = np.radians(input_data['pickup_latitude']), np.radians(input_data['pickup_longitude'])
    lat2, lon2 = np.radians(input_data['dropoff_latitude']), np.radians(input_data['dropoff_longitude'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    input_data['distance'] = 6371 * c
    input_data['distance'] = input_data['distance'].clip(upper=100)

    input_data['manhattan_distance'] = (
        np.abs(input_data['dropoff_latitude'] - input_data['pickup_latitude']) +
        np.abs(input_data['dropoff_longitude'] - input_data['pickup_longitude'])
    ) * 111
    input_data['manhattan_distance'] = input_data['manhattan_distance'].clip(upper=100)

    input_data['distance_per_hour'] = input_data['distance'] / (input_data['hour'] + 1)

    def get_time_of_day(hour):
        if 5 <= hour < 10:
            return 'morning'
        elif 10 <= hour < 16:
            return 'midday'
        elif 16 <= hour < 20:
            return 'evening'
        elif 20 <= hour < 24:
            return 'night'
        else:
            return 'late_night'
    input_data['time_of_day'] = input_data['hour'].apply(get_time_of_day)

    input_data['rush_hour'] = input_data['hour'].isin([7,8,9,10,16,17,18,19]).astype(int)
    input_data['weekend'] = (input_data['dayofweek'] >= 5).astype(int)

    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                        'dropoff_longitude', 'distance', 'manhattan_distance', 'distance_per_hour']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear',
                           'passenger_count', 'time_of_day', 'weekend', 'rush_hour']
    features = categorical_features + numeric_features
    
    return input_data[features]

model = load_model()
if model is None:
    st.stop()

st.header("Predict Trip Duration")

st.sidebar.header("Trip Details")
pickup_datetime = st.sidebar.text_input("Pickup Datetime (YYYY-MM-DD HH:MM:SS)", "2016-06-25 19:28:52")
pickup_latitude = st.sidebar.number_input("Pickup Latitude", value=40.763633728027344, format="%.6f")
pickup_longitude = st.sidebar.number_input("Pickup Longitude", value=-73.9763412475586, format="%.6f")
dropoff_latitude = st.sidebar.number_input("Dropoff Latitude", value=40.7434196472168, format="%.6f")
dropoff_longitude = st.sidebar.number_input("Dropoff Longitude", value=-73.97334289550781, format="%.6f")
passenger_count = st.sidebar.number_input("Passenger Count", min_value=1, max_value=6, value=1)

if st.sidebar.button("Predict Trip Duration"):
    try:
        datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
        
        input_features = prepare_input(
            pickup_datetime,
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
            passenger_count
        )
        
        log_prediction = model.predict(input_features)[0]
        prediction = np.expm1(log_prediction)
        
        st.success(f"**Estimated Trip Duration: {prediction:.2f} seconds (~{prediction/60:.2f} minutes)")
    except ValueError:
        st.error("Please enter a valid datetime format (YYYY-MM-DD HH:MM:SS).")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown("Built with ❤️ by Ahmed Ayman, Omar Khaled, and Ahmed Hassan")