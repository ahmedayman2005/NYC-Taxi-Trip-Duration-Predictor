import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

TRAIN_DATASET = 'train.csv'
TEST_DATASET = 'val.csv'

def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    mae = mean_absolute_error(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f} - MAE = {mae:.4f}")
    return rmse, r2, mae

def add_distance_features(df):
    df = df.copy()
    
    lat1, lon1 = np.radians(df['pickup_latitude']), np.radians(df['pickup_longitude'])
    lat2, lon2 = np.radians(df['dropoff_latitude']), np.radians(df['dropoff_longitude'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df['distance'] = 6371 * c 
    df['distance'] = df['distance'].clip(upper=100)

  
    df['manhattan_distance'] = (
        np.abs(df['dropoff_latitude'] - df['pickup_latitude']) +
        np.abs(df['dropoff_longitude'] - df['pickup_longitude'])
    ) * 111
    df['manhattan_distance'] = df['manhattan_distance'].clip(upper=100)

    
    df['distance_per_hour'] = df['distance'] / (df['hour'] + 1)

    
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
    df['time_of_day'] = df['hour'].apply(get_time_of_day)

   
    df['rush_hour'] = df['hour'].isin([7, 8, 9, 10, 16, 17, 18, 19]).astype(int)

  
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)

    return df

def approach1(train, test):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                        'dropoff_longitude', 'distance', 'manhattan_distance', 'distance_per_hour']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear',
                            'passenger_count', 'time_of_day', 'weekend', 'rush_hour']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ('scaling_poly', Pipeline([
            ('scaling', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features)
    ], remainder='passthrough')

    param_grid = {
        'regression__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    }

    def custom_rmse_scorer(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return -rmse

    distance_pipeline = Pipeline([
        ('feature_engineering', column_transformer),
        ('regression', Ridge(random_state=42))
    ])

    grid_search = RandomizedSearchCV(
        distance_pipeline,
        param_distributions=param_grid,
        n_iter=8,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring=make_scorer(custom_rmse_scorer),
        random_state=42,
        error_score=np.nan,
        return_train_score=True
    )

  
    sample_size = min(20000, len(train))
    sample_indices = np.random.choice(len(train), sample_size, replace=False)
    train_sample = train.iloc[sample_indices]

    print("Applying feature engineering...")
    train_sample = add_distance_features(train_sample)

    print("Starting grid search with expanded parameter space...")
    try:
        import time
        start_time = time.time()
        grid_search.fit(train_sample[train_features], train_sample.log_trip_duration)
        print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
        print("\n=== SEARCH COMPLETED ===")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV score:", grid_search.best_score_)
        best_params = grid_search.best_params_
        print("Applying feature engineering to full datasets...")
        train = add_distance_features(train)
        test = add_distance_features(test)
        best_ridge = Ridge(**{k.replace('regression__', ''): v
                              for k, v in best_params.items()},
                           random_state=42)
        final_pipeline = Pipeline([
            ('feature_engineering', column_transformer),
            ('regression', best_ridge)
        ])
        print("Fitting final model on full training data...")
        model = final_pipeline.fit(train[train_features], train.log_trip_duration)
    except Exception as e:
        print(f"Error during grid search: {e}")
        print("Falling back to default Ridge regressor")
        train = add_distance_features(train)
        test = add_distance_features(test)
        model = Pipeline(steps=[
            ('feature_engineering', column_transformer),
            ('regression', Ridge(alpha=1.0, random_state=42))
        ]).fit(train[train_features], train.log_trip_duration)

    train_rmse, train_r2, train_mae = predict_eval(model, train, train_features, "train")
    test_rmse, test_r2, test_mae = predict_eval(model, test, train_features, "test")

    return model, train_rmse, train_r2, train_mae, test_rmse, test_r2, test_mae

def prepare_data(train):
    train = train.copy()
    train.drop(columns=['id'], errors='ignore', inplace=True)
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dayofweek'] = train.pickup_datetime.dt.dayofweek
    train['month'] = train.pickup_datetime.dt.month
    train['hour'] = train.pickup_datetime.dt.hour
    train['dayofyear'] = train.pickup_datetime.dt.dayofyear
    train['log_trip_duration'] = np.log1p(train.trip_duration)

    train = train.dropna(subset=['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                                'dropoff_latitude', 'dropoff_longitude', 'trip_duration'])

    train = train[train['trip_duration'].between(30, 24*3600)]
    train = train[train['pickup_latitude'].between(40.6, 40.9) & 
                  train['pickup_longitude'].between(-74.1, -73.7) &
                  train['dropoff_latitude'].between(40.6, 40.9) & 
                  train['dropoff_longitude'].between(-74.1, -73.7)]

    if train.empty:
        raise ValueError("Training data is empty after preprocessing")

    return train

train = pd.read_csv(TRAIN_DATASET)
test = pd.read_csv(TEST_DATASET)
train = prepare_data(train)
test = prepare_data(test)

approach1_model, approach1_train_rmse, approach1_train_r2, approach1_train_mae, approach1_test_rmse, approach1_test_r2, approach1_test_mae = approach1(train, test)

import pickle
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'approach1_model_ridge_20000_fixed.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(approach1_model, f)
print(f"Model saved to {model_path}")

def predict_trip_duration(pickup_datetime, pickup_latitude, pickup_longitude,
                          dropoff_latitude, dropoff_longitude, passenger_count,
                          model_path='models/approach1_model_ridge_20000_fixed.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    input_data = pd.DataFrame({
        'pickup_datetime': [pickup_datetime],
        'pickup_latitude': [pickup_latitude],
        'pickup_longitude': [pickup_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'passenger_count': [passenger_count]
    })

    input_data['pickup_datetime'] = pd.to_datetime(input_data['pickup_datetime'])
    input_data['dayofweek'] = input_data.pickup_datetime.dt.dayofweek
    input_data['month'] = input_data.pickup_datetime.dt.month
    input_data['hour'] = input_data.pickup_datetime.dt.hour
    input_data['dayofyear'] = input_data.pickup_datetime.dt.dayofyear
    input_data = add_distance_features(input_data)

    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                        'dropoff_longitude', 'distance', 'manhattan_distance', 'distance_per_hour']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear',
                            'passenger_count', 'time_of_day', 'weekend', 'rush_hour']
    features = categorical_features + numeric_features

    log_prediction = model.predict(input_data[features])[0]
    prediction = np.expm1(log_prediction)
    return prediction

predicted_duration = predict_trip_duration(
    pickup_datetime='2016-06-25 19:28:52',
    pickup_latitude=40.763633728027344,
    pickup_longitude=-73.9763412475586,
    dropoff_latitude=40.7434196472168,
    dropoff_longitude=-73.97334289550781,
    passenger_count=1
)
print(f"Predicted trip duration: {predicted_duration:.2f} seconds")