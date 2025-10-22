"""Preprocessing Template untuk Latihan Data Mining (sesuai slide)
- Siapkan file Data.csv (letakkan di folder yang sama)
- Script ini melakukan: load data, inspect, handle missing values,
  encoding kategori, split train-test, dan feature scaling.
- Menyimpan hasil preprocessing ke 'processed_data.csv'.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def main():
    # Ganti nama file jika perlu
    filename = 'Data.csv'

    # 1) Load dataset
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File '{filename}' tidak ditemukan. Silakan tempatkan Data.csv di folder yang sama.")
        return

    print('\n=== INFO DATA ===')
    print(df.info())
    print('\n=== 5 BARIS PERTAMA ===')
    print(df.head())

    # 2) Contoh deteksi missing value dan ringkasan
    print('\n=== RINGKASAN MISSING VALUE ===')
    print(df.isnull().sum())

    # 3) Menghapus kolom yang tidak relevan (misal: 'Name' atau 'ID' jika ada)
    for col in ['Name', 'ID', 'NIM']:
        if col in df.columns:
            print(f"Menghapus kolom tidak relevan: {col}")
            df.drop(columns=[col], inplace=True)

    # 4) Imputasi missing value untuk kolom numerik menggunakan mean
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
        print(f"Imputasi mean untuk kolom numerik: {num_cols}")

    # 5) Imputasi missing value untuk kolom kategorikal menggunakan modus
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
        print(f"Imputasi modus untuk kolom kategorikal: {cat_cols}")

    # 6) Encoding label target (jika bernama 'target' atau 'Lulus_tepat_waktu')
    target_candidates = [c for c in df.columns if c.lower() in ['target','lulus_tepat_waktu','kelulusan','status']]
    if target_candidates:
        target = target_candidates[0]
        print(f"Menemukan kolom target: {target}")
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
    else:
        print('Tidak menemukan kolom target otomatis. Pastikan ada kolom target/label.')
        target = None

    # 7) One-hot encoding untuk kolom kategorikal selain target
    cat_cols = [c for c in cat_cols if c != target] if cat_cols and target else cat_cols
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"One-hot encoding untuk: {cat_cols}")

    # 8) Split X and y
    if target:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        X = df.copy()
        y = None

    # 9) Split train-test jika ada target
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Ukuran train: {X_train.shape}, ukuran test: {X_test.shape}")
    else:
        print('Tidak melakukan split karena tidak ada target.')

    # 10) Feature scaling untuk fitur numerik
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_features:
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        print(f"Feature scaling (StandardScaler) untuk: {numeric_features}")

    # 11) Menyimpan hasil preprocessing
    out_path = 'processed_data.csv'
    df.to_csv(out_path, index=False)
    print(f"Hasil preprocessing disimpan di: {out_path}")

if __name__ == '__main__':
    main()
