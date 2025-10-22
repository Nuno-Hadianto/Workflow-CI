import pandas as pd
import mlflow
from mlflow.sklearn import save_model 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import shutil # Untuk menghapus direktori

# Matikan warnings
warnings.filterwarnings("ignore")

# Path ke data
DATA_PATH = os.path.join("namadataset_preprocessing", "netflix_preprocessed.csv")
# Path untuk menyimpan run_id
RUN_ID_FILE = "run_id.txt"
# Direktori lokal sementara untuk menyimpan model
LOCAL_MODEL_DIR = "local_model_output" 

print("Memulai script training CI (versi hardcoded workflow)...")

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# 2. Feature Engineering dan Encoding
df['features_text'] = df['title'].fillna('') + ' ' + \
                      df['director'].fillna('') + ' ' + \
                      df['cast'].fillna('') + ' ' + \
                      df['listed_in'].fillna('') + ' ' + \
                      df['description'].fillna('')
le = LabelEncoder()
y = le.fit_transform(df['type'])
X = df['features_text']
print("Feature engineering selesai.")

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Membuat dan Melatih Pipeline Model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
    ('model', LogisticRegression(random_state=42, max_iter=1000, C=10.0))
])

print("Memulai training model...")

# 5. MLflow Tracking
try:
    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        print("Training model selesai.")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Log parameter
        mlflow.log_param("C", 10.0)
        mlflow.log_param("max_features", 3000)
        
        # Log metrik
        accuracy = pipeline.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Akurasi: {accuracy}")

        # --- LOG MODEL (save_model + log_artifacts) ---
        print(f"Menyimpan model ke direktori lokal: {LOCAL_MODEL_DIR}...")
        # Hapus direktori lama jika ada
        if os.path.exists(LOCAL_MODEL_DIR):
            shutil.rmtree(LOCAL_MODEL_DIR)
        # Simpan model dengan format MLflow (membuat MLmodel, conda.yaml, dll.)
        save_model(pipeline, LOCAL_MODEL_DIR)
        print("Model berhasil disimpan secara lokal.")

        print(f"Mengunggah artefak dari {LOCAL_MODEL_DIR} ke DagsHub path 'model'...")
        # Unggah seluruh isi direktori ke path 'model' di DagsHub
        mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="model")
        print("Artefak model berhasil diunggah.")
        
        # Hapus direktori lokal setelah diunggah
        shutil.rmtree(LOCAL_MODEL_DIR)
        # --- SELESAI LOG MODEL BARU ---

        # 6. Simpan run_id ke file
        with open(RUN_ID_FILE, "w") as f:
            f.write(run_id)
        print(f"Run ID disimpan ke {RUN_ID_FILE}")
except Exception as e:
    print(f"\n!!! TERJADI ERROR SAAT TRAINING/LOGGING !!!")
    print(f"Error: {e}")
    # Bersihkan direktori lokal jika error terjadi sebelum sempat dihapus
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)
    exit(1)
    
print("Script training CI selesai.")
