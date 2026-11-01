
#For data manipulation

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# For model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# For model serialization
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:8888")
mlflow.set_experiment("mlops-training-experiment")

# X_train/X_test/y_train/y_test are downloaded from previous job's artifact now need to read
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_read_csv("y_test.csv").squeeze()

# One-hot encode and scale numeric feature'

# Numerical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()

# Set class weight to handle class imbalance

class_weight = y_train.value_counts()[0]/y_train.value_counts()[1]

#Define the preprocessing steps

preprocessing = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown='ignore'), cat_cols)
)

#Define base XGBoost model

xgb_model = xgb.XGBRFClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# Define hyperparameter grid and limiting the parameter to run to google colab environment

param_grid = {
    'xgbrfclassifier__n_estimators': [50,100],
    'xgbrfclassifier__max_depth': [2,3],
    'xgbrfclassifier__learning_rate': [0.05, 0.1],
    #'xgbrfclassifier_colsample_bytree': [0.4, 0.5, 0.6],
    #'xgbrfclassifier_colsample_bylevel': [0.4, 0.5, 0.6],
    #'xgbrfclassifier_reg_lambda': [0.4, 0.5, 0.6],
}

#Model pipeline

model_pipeline = make_pipeline(preprocessing, xgb_model)

with mlflow.start_run():
  # Hyper parameter tuning
  grid_search = GridSearchCV(model_pipeline, param_grid, cv=5,
  scoring = 'recall', n_jobs=-1)
  grid_search.fit(X_train, y_train)

  # Log all parameter combinations and their mean test score
  result = grid_search.cv_results_
  for i in range(len(result['params'])):
    param_set = result['params'][i]
    mean_score = result["mean_test_score"][i]
    std_score = result["std_test_score"][i]

    #Log each combinations as a separate MLflow run
    with mlflow.start_run(nested=True):
      mlflow.log_params(param_set)
      mlflow.log_metric("mean_test_score", mean_score)
      mlflow.log_metric("std_test_score", std_score)
  
  # Log best parameters separately in main run
  mlflow.log_params(grid_search.best_params_)
  
  # Store and evaluate the best model
  best_model = grid_search.best_estimator_
  print("Best params:",grid_search.best_params_)
  
  classification_threshold = 0.45

  y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
  y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
  
  y_pred_test_proba =best_model.predict_proba(X_test)[:, 1]
  y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)
  
  train_report = classification_report(y_train, y_pred_train,output_dict=True)
  test_report = classification_report(y_test, y_pred_test,output_dict=True)  
  
  print(classification_report(y_train, y_pred_train))
  print(classification_report(y_test, y_pred_test)
  
  mlflow.log_metrics({
      "train_accuracy": train_report["accuracy"],
      "train_precision": train_report["weighted avg"]["precision"],
      "train_recall": train_report["weighted avg"]["recall"],
      "train_f1_score": train_report["weighted avg"]["f1-score"],
      "test_accuracy": test_report["accuracy"],
      "test_precision": test_report["weighted avg"]["precision"],
      "test_recall": test_report["weighted avg"]["recall"],
      "test_f1_score": test_report["weighted avg"]["f1-score"],
  })

  # Save next to app.py so the Strealit app can load it directly, and log
  # it as an MLflow artificate for traceability

  model_path = 'tourism_project/deployment/best_tourism_package_prediction_model.joblib'
  joblib.dump(best_model, model_path)
  mlflow.log_artifact(model_path,artifact_path='model')
  print(f'Model saved to {model_path}')      
