# CaloriQ — AI-Powered Calorie Prediction

CaloriQ is a machine learning-based fitness analytics application that predicts calories burned during exercise using user-provided fitness and workout information.

## Features
- Predict calories burned from workout and body measurements.
- Compare six regression models.
- Evaluate models using MAE, MSE, RMSE, and R².
- Visualize model performance and feature importance.
- Explore fitness data through a Streamlit dashboard.

## Machine Learning Models
- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Support Vector Regression (SVR)

## Tech Stack
Python, Streamlit, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, and Seaborn.

## Run Locally

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Ensure the trained model file is available.
4. Run `streamlit run app.py`.

The repository includes `runtime.txt` and `.python-version` to keep deployments on
Python 3.10, which matches the serialized scikit-learn model. For Streamlit
Community Cloud, select this repository and use `app.py` as the entrypoint.

## Disclaimer
Calorie predictions are estimates for educational and informational purposes, not medical advice.
