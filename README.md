# 🌧️ Rainfall Prediction in Australia: An End-to-End ML Pipeline

## 📋 Project Overview
This project demonstrates a complete machine learning workflow to predict daily rainfall in the Melbourne region using historical data from the **Australian Bureau of Meteorology**. 

The core challenge of this project was navigating **Temporal Data Leakage** and handling an **imbalanced dataset** (where rainy days are the minority) to build a model that provides meaningful real-world utility.

## 🛠️ Technical Stack
* **Language:** Python 3.12
* **Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
* **Models:** Random Forest Classifier, Logistic Regression
* **Optimization:** GridSearchCV with 5-fold Cross-Validation

---

## 🚀 Key Machine Learning Workflows

### 1. Feature Engineering & Data Cleaning
* **Seasonal Logic:** Converted raw `Date` strings into `Season` categories (Summer, Autumn, Winter, Spring) to capture cyclical weather patterns.
* **Target Shifting:** To prevent "cheating" (Data Leakage), I shifted the timeline to predict **Today's** rain using **Yesterday's** finalized observations. This ensures the model only uses data that would actually be available at 9:00 AM on the day of the forecast.

### 2. Automated Preprocessing Pipeline
Using Scikit-Learn's `ColumnTransformer`, I built a unified pipeline to handle diverse data types:
* **Numerical Features:** Applied `StandardScaler` to normalize features like Pressure, Temperature, and Wind Speed.
* **Categorical Features:** Applied `OneHotEncoder` to handle Location, Wind Direction, and Seasons.

### 3. Hyperparameter Tuning
I used `GridSearchCV` to exhaustively test model configurations:
* **Random Forest:** Tuned `n_estimators` and `max_depth` to prevent overfitting.
* **Logistic Regression:** Optimized `penalty` types and used `class_weight='balanced'` to give more importance to the minority "Rainy" class.

---

## 📊 Performance Results
The test set contained 1,512 observations with a significant class imbalance (~76% No Rain / ~24% Rain).

| Metric | Random Forest | Logistic Regression |
| :--- | :--- | :--- |
| **Accuracy** | **84%** | 83% |
| **True Positive Rate (Recall)** | **51%** | 51% |
| **Precision (Rain)** | **75%** | 68% |
| **F1-Score (Rain)** | **0.61** | 0.58 |

**Analysis:** While both models struggled to catch every rainy day (51% Recall), the **Random Forest** proved to be the superior predictor due to its significantly higher **Precision**, resulting in far fewer "False Alarms."

---

## 🔍 Feature Importance
The Random Forest model identified the following as the strongest predictors for rainfall:
1. **Humidity3pm** (Highest Impact)
2. **Pressure3pm**
3. **Pressure9am**
4. **Sunshine**

---

## 📂 Repository Structure
* `notebooks/`: Contains the Jupyter Notebook with the full analysis.
* `data/`: (Optional) Link to the [official Bureau of Meteorology source](http://www.bom.gov.au/climate/dwo/).
* `requirements.txt`: List of dependencies to reproduce the environment.
* `README.md`: Project documentation.

## 📈 Future Work
* **Threshold Adjustment:** Moving the classification threshold from 0.5 to 0.3 to prioritize Recall (catching more rain events).
* **Advanced Ensembles:** Implementing XGBoost or LightGBM to improve non-linear pattern recognition.
* **SMOTE:** Using Synthetic Minority Over-sampling to improve the model's ability to learn from rainy days.
