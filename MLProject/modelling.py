import pandas as pd
import numpy as np
import time
import os
import joblib
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# --- MLflow Setup ---
# Import dagshub dan mlflow tetap diperlukan untuk MLflow Tracking
import dagshub 
import mlflow 

# Abaikan warning deprecation dari scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(path: str):
    """Memuat data pelatihan dan pengujian yang sudah diproses."""
    train_df = pd.read_csv(f'{path}/train_processed.csv')
    test_df = pd.read_csv(f'{path}/test_processed.csv')
    
    X_train = train_df.drop('Salary', axis=1)
    y_train = train_df['Salary']
    X_test = test_df.drop('Salary', axis=1)
    y_test = test_df['Salary']
    
    return X_train, X_test, y_train, y_test


def train_and_log_model(X_train, X_test, y_train, y_test):
    """Melatih, tuning, dan mencatat model ke MLflow."""
    
    # 1. Definisikan Algoritma & Hyperparameter Tuning
    model = Ridge(random_state=42)
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 50.0], 
        'solver': ['cholesky', 'lsqr', 'svd'] 
    }
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        error_score=0 # Mengabaikan fit yang gagal
    )
    
    start_time = time.time()
    
    # 2. Start MLflow Run 
    with mlflow.start_run(run_name="Ridge_Tuning_Advanced"):
        print("Memulai pelatihan dan tuning...")
        grid_search.fit(X_train, y_train)
        
        end_time = time.time()
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # 3. Hitung Metrik Evaluasi
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        # 4. Manual Logging (Kriteria 2: Skilled/Advance)
        print("Mulai Manual Logging...")
        
        # Log Hyperparameters terbaik
        mlflow.log_params(grid_search.best_params_)
        
        # Log Metrik Utama
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Log Metrik Tambahan (Minimal 2)
        mlflow.log_metric("time_to_train_sec", end_time - start_time)
        mlflow.log_metric("num_features", X_train.shape[1])
        mlflow.log_metric("data_rows_used", len(X_train) + len(X_test)) 
        
        # --- LOGGING ARTEFAK (Kriteria 2: Advance) ---
        TEMP_DIR = "temp_model_artifact"
        os.makedirs(TEMP_DIR, exist_ok=True)
        MODEL_FILE_NAME = "best_ridge_model.pkl"

        # Simpan model menggunakan joblib
        joblib.dump(best_model, os.path.join(TEMP_DIR, MODEL_FILE_NAME))
        
        # Log model ke MLflow
        mlflow.log_artifact(TEMP_DIR, artifact_path="ridge_model") 
        
        # Hapus folder sementara
        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            pass
        
        print(f"Pelatihan selesai. Metrik disimpan di DagsHub: RMSE={rmse:.2f}")

        
if __name__ == '__main__':
    PREPROCESSED_DATA_PATH = 'namadataset_preprocessing' 
    
    X_train, X_test, y_train, y_test = load_data(PREPROCESSED_DATA_PATH)
    train_and_log_model(X_train, X_test, y_train, y_test)
