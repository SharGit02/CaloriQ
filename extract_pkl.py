"""
extract_pkl.py
Robust script to extract and inspect the model from 'fitstats_calorie_model (1).pkl',
compute model metrics, feature importances, comparison of 6 regression models on
gym_members_exercise_tracking.csv, and export everything cleanly to model_data.json.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Sklearn Cross-Version Compatibility Patch ─────────────────────────────────
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            def __reduce__(self):
                return (list, (list(self),))
        _ct._RemainderColsList = _RemainderColsList
except Exception as e:
    pass

import joblib

def safe_convert(obj, depth=0):
    """Recursively convert NumPy and complex types to JSON-serializable types."""
    if depth > 6:
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): safe_convert(v, depth + 1) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_convert(i, depth + 1) for i in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

PKL_FILE = 'fitstats_calorie_model (1).pkl'
CSV_FILE = 'gym_members_exercise_tracking.csv'

print(f"Loading '{PKL_FILE}'...")

data = None
# Strategy 1: Joblib (the pkl was serialized using joblib)
try:
    data = joblib.load(PKL_FILE)
    print("SUCCESS: Loaded model using joblib!")
except Exception as ej:
    print(f"Joblib load failed ({ej}), attempting pickle...")
    # Strategy 2: Standard pickle with fallback
    import pickle
    try:
        with open(PKL_FILE, 'rb') as f:
            data = pickle.load(f)
        print("SUCCESS: Loaded model using standard pickle!")
    except Exception as ep:
        print(f"Standard pickle failed: {ep}")
        # Strategy 3: Compat unpickler
        class CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    class Dummy:
                        def __init__(self, *args, **kwargs): pass
                        def __setstate__(self, state):
                            if isinstance(state, dict):
                                self.__dict__.update(state)
                    Dummy.__name__ = name
                    Dummy.__module__ = module
                    return Dummy
        with open(PKL_FILE, 'rb') as f:
            data = CompatUnpickler(f).load()
        print("Loaded with CompatUnpickler fallback.")

print("Loaded object type:", type(data))

output = {
    "model_type": type(data).__name__,
    "module": getattr(type(data), '__module__', ''),
    "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Check if data is dict or Pipeline/Estimator
if isinstance(data, dict):
    print("Detected dictionary structure. Extracting keys...")
    for k, v in data.items():
        output[k] = safe_convert(v)
else:
    # It's an Estimator or Pipeline
    print("Detected scikit-learn Pipeline / Estimator structure.")
    
    # 1. Pipeline steps
    if hasattr(data, 'named_steps'):
        output['steps'] = list(data.named_steps.keys())
        for step_name, step_obj in data.named_steps.items():
            output[f'step_{step_name}_type'] = type(step_obj).__name__
            for attr in ['feature_names_in_', 'n_features_in_']:
                if hasattr(step_obj, attr):
                    output[f'step_{step_name}_{attr}'] = safe_convert(getattr(step_obj, attr))
            
            # Check model in step
            if hasattr(step_obj, 'feature_importances_'):
                output['feature_importances_'] = safe_convert(step_obj.feature_importances_)
            if hasattr(step_obj, 'get_params'):
                try:
                    output[f'step_{step_name}_params'] = safe_convert(step_obj.get_params())
                except Exception:
                    pass

    # Extract feature names if preprocessor exists
    feature_names = []
    if hasattr(data, 'named_steps') and 'preprocessor' in data.named_steps:
        prep = data.named_steps['preprocessor']
        try:
            feature_names = list(prep.get_feature_names_out())
            output['feature_names_out'] = feature_names
        except Exception:
            pass

    # Extract feature importances mapped to feature names
    if hasattr(data, 'named_steps') and 'model' in data.named_steps:
        estimator = data.named_steps['model']
        if hasattr(estimator, 'feature_importances_'):
            fi = estimator.feature_importances_
            output['feature_importances_raw'] = safe_convert(fi)
            if feature_names and len(feature_names) == len(fi):
                fi_mapped = sorted(
                    [{"feature": fn, "importance": float(imp)} for fn, imp in zip(feature_names, fi)],
                    key=lambda x: x["importance"],
                    reverse=True
                )
                output['feature_importances_ranked'] = fi_mapped
                print(f"\nTop 5 Most Important Features:")
                for item in fi_mapped[:5]:
                    print(f"  - {item['feature']}: {item['importance']:.4f}")

    # 2. Evaluate on gym_members_exercise_tracking.csv if present
    if os.path.exists(CSV_FILE):
        print(f"\nFound dataset '{CSV_FILE}'. Evaluating model performance & comparing 6 algorithms...")
        try:
            df = pd.read_csv(CSV_FILE)
            output['dataset_shape'] = list(df.shape)
            output['dataset_columns'] = list(df.columns)
            
            target_col = 'Calories_Burned'
            if target_col in df.columns:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import LinearRegression, Ridge
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.svm import SVR
                
                # Split features & target
                feature_cols = [c for c in df.columns if c != target_col]
                X = df[feature_cols]
                y = df[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Predict using loaded pipeline
                y_pred = data.predict(X_test)
                mae = float(mean_absolute_error(y_test, y_pred))
                mse = float(mean_squared_error(y_test, y_pred))
                rmse = float(np.sqrt(mse))
                r2 = float(r2_score(y_test, y_pred))
                
                output['pipeline_evaluation'] = {
                    'MAE': round(mae, 3),
                    'MSE': round(mse, 2),
                    'RMSE': round(rmse, 3),
                    'R2_Score': round(r2, 4)
                }
                print(f"Pipeline Evaluation -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
                
                # Sample predictions for actual vs predicted
                output['sample_actual_vs_predicted'] = [
                    {"actual": float(a), "predicted": round(float(p), 1)}
                    for a, p in zip(y_test.values[:30], y_pred[:30])
                ]
                
                # Compare 6 Regression Models
                prep = data.named_steps.get('preprocessor')
                from sklearn.pipeline import Pipeline
                
                models = {
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(alpha=1.0),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Support Vector Regressor": SVR(C=100.0, epsilon=0.1)
                }
                
                comparison = []
                for name, reg in models.items():
                    pipe = Pipeline([('preprocessor', prep), ('model', reg)])
                    t0 = time.time()
                    pipe.fit(X_train, y_train)
                    t_fit = time.time() - t0
                    y_p = pipe.predict(X_test)
                    m_mae = float(mean_absolute_error(y_test, y_p))
                    m_mse = float(mean_squared_error(y_test, y_p))
                    m_rmse = float(np.sqrt(m_mse))
                    m_r2 = float(r2_score(y_test, y_p))
                    comparison.append({
                        "Model": name,
                        "MAE": round(m_mae, 2),
                        "MSE": round(m_mse, 1),
                        "RMSE": round(m_rmse, 2),
                        "R2": round(m_r2, 4),
                        "Training_Time_s": round(t_fit, 3)
                    })
                
                comparison.sort(key=lambda x: x["R2"], reverse=True)
                output['model_comparison'] = comparison
                print("\n6 Models Comparison:")
                for m in comparison:
                    print(f"  {m['Model']:<25} | MAE: {m['MAE']:<6} | RMSE: {m['RMSE']:<6} | R2: {m['R2']:<6} | Time: {m['Training_Time_s']}s")
        except Exception as ed:
            print(f"Error during dataset evaluation: {ed}")
            output['evaluation_error'] = str(ed)

# Save to model_data.json
with open('model_data.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("\nSuccessfully saved all extracted metadata and analytics to 'model_data.json'!")
print(f"Summary of keys in model_data.json: {list(output.keys())}")
