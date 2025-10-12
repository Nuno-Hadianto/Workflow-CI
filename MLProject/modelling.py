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
import mlflow 

warnings.filterwarnings("ignore", category=FutureWarning)

# Catatan: dagshub.init() DIHAPUS. Autentikasi dilakukan oleh GitHub Secrets.

def load_data(): # Hapus parameter 'path' karena kita menggunakan path absolut
    """Memuat data pelatihan dan pengujian yang sudah diproses."""
    
    # PERBAIKAN PATH DATA: Menggunakan path absolut dari root repository
    # Format: [Nama Folder MLProject]/[Nama Folder Preprocessing]/[Nama File]
    TRAIN_PATH = 'MLProject/namadataset_preprocessing/train_processed.csv'
    TEST_PATH = 'MLProject/namadataset_preprocessing/test_processed.csv'
    
    # Cek apakah file ada (untuk debug)
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"File data tidak ditemukan! Periksa struktur folder CI Anda. Harusnya ada di: {TRAIN_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop('Salary', axis=1)
    y_train = train_df['Salary']
    X_test = test_df.drop('Salary', axis=1)
    y_test = test_df['Salary']
    
    return X_train, X_test, y_train, y_test


def train_and_log_model(X_train, X_test, y_train, y_test):
    """Melatih, tuning, dan mencatat model ke MLflow."""
    
    # ... (Sisa kode fungsi training Anda, TANPA PERUBAHAN)
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
        error_score=0 
    )
    
    start_time = time.time()
    
    with mlflow.start_run(run_name="Ridge_Tuning_CI"):
        print("Memulai pelatihan dan tuning...")
        grid_search.fit(X_train, y_train)
        
        end_time = time.time()
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Log Metrik (Kriteria 2: Advance)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("time_to_train_sec", end_time - start_time)
        mlflow.log_metric("num_features", X_train.shape[1])
        mlflow.log_metric("data_rows_used", len(X_train) + len(X_test)) 
        
        # --- LOGGING ARTEFAK YANG STABIL ---
        TEMP_DIR = "temp_model_artifact"
        os.makedirs(TEMP_DIR, exist_ok=True)
        MODEL_FILE_NAME = "best_ridge_model.pkl"

        joblib.dump(best_model, os.path.join(TEMP_DIR, MODEL_FILE_NAME))
        
        mlflow.log_artifact(TEMP_DIR, artifact_path="ridge_model") 
        
        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            pass
        
        print(f"Pelatihan selesai. Metrik disimpan di DagsHub: RMSE={rmse:.2f}")

        
if __name__ == '__main__':
    # Tidak perlu passing path lagi
    X_train, X_test, y_train, y_test = load_data() 
    train_and_log_model(X_train, X_test, y_train, y_test)
