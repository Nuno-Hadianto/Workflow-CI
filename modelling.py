import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
import os

# Matikan warnings
warnings.filterwarnings("ignore")

# Path ke data
DATA_PATH = os.path.join("namadataset_preprocessing", "netflix_preprocessed.csv")
# Path untuk menyimpan model
MODEL_OUTPUT_DIR = "model"
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "model.pkl")

print("Memulai script training CI...")

# 1. Load Data
print(f"Memuat data dari {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset tidak ditemukan di {DATA_PATH}")
    exit()

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
# parameter terbaik dari Kriteria 2
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
    ('model', LogisticRegression(random_state=42, max_iter=1000, C=10.0))
])

print("Memulai training model...")
pipeline.fit(X_train, y_train)
print("Training model selesai.")

# 5. Menyimpan Model sebagai Artefak
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"Menyimpan model ke {MODEL_PATH}...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Model berhasil disimpan di {MODEL_PATH}.")
print("Script training CI selesai.")
