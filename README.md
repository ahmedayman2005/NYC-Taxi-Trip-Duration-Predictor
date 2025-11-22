#   NYC Taxi Trip Duration Predictor

---

##  Project Overview

This project predicts the duration of NYC taxi trips using machine learning.  
It uses a Ridge Regression model combined with advanced feature engineering to analyze pickup/dropoff locations, time of day, rush hour effects, and distance metrics.  
The project also includes an interactive Streamlit application for real-time trip duration predictions.

The project includes:

- Full preprocessing & data cleaning  
- Advanced feature engineering  
- Hyperparameter tuning with RandomizedSearchCV  
- Model training & evaluation  
- Streamlit deployment  
- Final performance metrics (Train, Validation, Test)  

---

##  Project Files

| File | Description |
|------|-------------|
| `NYC Trip Duration.py` | Full pipeline for feature engineering and model training |
| `app.py` | Streamlit web app for live trip duration predictions |
| `model.pkl` | Trained Ridge Regression model |
| `requirements.txt` | Python dependencies |
| `models/` | Saved model versions |

---

##  Data Preparation Summary

This project applies several cleaning and preprocessing steps:

- Removed trips with unrealistic durations  
- Filtered impossible GPS coordinates  
- Log-transformed the target variable (trip duration)  
- Extracted time-based features from the pickup timestamp  
- Generated distance features:
  - **Haversine distance**
  - **Manhattan distance**
  - **Distance per hour**
- Created category features:
  - **Rush hour flag**
  - **Weekend flag**
  - **Time-of-day segmentation** (Morning, Midday, Evening, Night)

---

##  Feature Engineering Details

Key engineered features improving prediction accuracy:

- **Haversine Distance** – real geographic distance  
- **Manhattan Distance** – grid-approximate NYC street distance  
- **Distance Per Hour** – speed normalization  
- **Time of Day** – categorical segmentation  
- **Rush Hour Flag** – (7–10 AM, 4–7 PM)  
- **Weekend Flag** – Saturday & Sunday indicator  

These features significantly improved the model's R² score.

---

##  Model Performance

###  Training Set
- **RMSE (log):** 0.4720  
- **R² (log):** 0.6126  
- **MAE (log):** 0.3356  

###  Validation Set
- **RMSE (log):** 0.4743  
- **R² (log):** 0.6111  
- **MAE (log):** 0.3362  

###  Test Set
- **RMSE (log):** 0.4697  
- **R² (log):** 0.6165  
- **MAE (log):** 0.3345  

The model generalizes well with strong consistency across all splits.

---

##  Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib / Pickle  

---

##  Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
