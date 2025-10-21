import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import pickle

# Matikan warnings
warnings.filterwarnings("ignore")

# Path ke data
DATA_PATH = os.path.join("namadataset_preprocessing", "netflix_preprocessed.csv")
# Path untuk menyimpan run_id
RUN_ID_FILE = "run_id.txt"

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

        # Log model ke DagsHub
        print("Melakukan logging model ke DagsHub...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model" 
        )
        print("Logging model berhasil.")

        # 6. Simpan run_id ke file
        with open(RUN_ID_FILE, "w") as f:
            f.write(run_id)
        print(f"Run ID disimpan ke {RUN_ID_FILE}")
except Exception as e:
    print(f"\n!!! TERJADI ERROR SAAT TRAINING/LOGGING !!!")
    print(f"Error: {e}")
    exit(1)
    
print("Script training CI selesai.")
