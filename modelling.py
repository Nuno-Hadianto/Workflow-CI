import mlflow
import os
import time

print("--- MEMULAI TES KONEKSI DAGSHUB ---")

# 1. Baca Secrets (Sama seperti sebelumnya)
uri = os.environ.get("DAGSHUB_URI")
username = os.environ.get("DAGSHUB_USERNAME")
token = os.environ.get("DAGSHUB_TOKEN")

if not uri or not username or not token:
    print("!!! ERROR: SECRETS DAGSHUB HILANG !!!")
    exit(1)

print("Secrets Ditemukan.")
print(f"URI: {uri}")
print(f"User: {username}")

# 2. Set Environment Variables MLflow
os.environ["MLFLOW_TRACKING_URI"] = uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# 3. Coba Mulai Run (Titik Potensi Error)
try:
    print("Mencoba mlflow.start_run()...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"BERHASIL! Run ID: {run_id}")
        mlflow.log_param("test_param", "success")
        print("Parameter 'test_param' berhasil di-log.")
        
        # Simpan run_id (masih diperlukan untuk langkah Docker)
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print("Run ID disimpan ke run_id.txt")

except Exception as e:
    print("\n!!! GAGAL MELAKUKAN mlflow.start_run() !!!")
    print(f"Error: {e}")
    print("Penyebab paling mungkin: Value/Isi dari DAGSHUB_URI, DAGSHUB_USERNAME, atau DAGSHUB_TOKEN masih salah.")
    print("Harap generate token BARU lagi, dan copy-paste ulang SEMUA secrets DagsHub di GitHub.")
    exit(1) # GAGALKAN SCRIPT

print("--- TES KONEKSI DAGSHUB SELESAI ---")
# Kita tidak melakukan training, jadi script selesai di sini.
# Langkah Docker di workflow akan tetap berjalan jika koneksi berhasil.
