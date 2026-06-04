import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FUNGSI-FUNGSI PREPROCESSING
# ============================================================

def load_data(filepath):
    """Load dataset dari file CSV."""
    print(f"[1] Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"    Shape awal: {df.shape}")
    return df


def handle_missing_values(df):
    """Tangani missing values."""
    print("[2] Menangani missing values...")
    
    # Age: isi dengan median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: isi dengan modus
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fare: isi dengan median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Cabin: terlalu banyak missing, drop kolom
    df = df.drop(columns=['Cabin'], errors='ignore')
    
    print(f"    Missing values tersisa: {df.isnull().sum().sum()}")
    return df


def drop_irrelevant_columns(df):
    """Hapus kolom yang tidak relevan untuk model."""
    print("[3] Menghapus kolom tidak relevan...")
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"    Kolom tersisa: {list(df.columns)}")
    return df


def encode_categorical(df):
    """Encode kolom kategorikal."""
    print("[4] Encoding kolom kategorikal...")
    le = LabelEncoder()
    
    # Sex: male=1, female=0
    df['Sex'] = le.fit_transform(df['Sex'])
    
    # Embarked: S=2, Q=1, C=0
    df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
    
    print(f"    Selesai encode: Sex, Embarked")
    return df


def feature_engineering(df):
    """Tambah fitur baru yang berguna."""
    print("[5] Feature engineering...")
    
    # FamilySize = SibSp + Parch + 1 (diri sendiri)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone: 1 jika tidak punya keluarga
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    print(f"    Fitur baru ditambahkan: FamilySize, IsAlone")
    return df


def scale_features(df):
    """Standarisasi fitur numerik."""
    print("[6] Scaling fitur numerik...")
    scaler = StandardScaler()
    cols_to_scale = ['Age', 'Fare', 'FamilySize']
    
    # Hanya scale kolom yang ada
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print(f"    Kolom yang di-scale: {cols_to_scale}")
    return df


def split_and_save(df, output_dir):
    """Pisahkan fitur & target, split train/test, simpan."""
    print("[7] Split data dan simpan...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Simpan semua versi
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    df.to_csv(os.path.join(output_dir, 'titanic_preprocessing.csv'), index=False)
    
    print(f"    Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"    File disimpan di: {output_dir}")
    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_preprocessing(input_path, output_dir):
    """Jalankan seluruh pipeline preprocessing."""
    print("=" * 50)
    print("  PIPELINE PREPROCESSING TITANIC")
    print("=" * 50)
    
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = drop_irrelevant_columns(df)
    df = encode_categorical(df)
    df = feature_engineering(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_and_save(df, output_dir)
    
    print("=" * 50)
    print("  PREPROCESSING SELESAI!")
    print("=" * 50)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT  = os.path.join(BASE_DIR, '..', 'titanic_raw', 'titanic.csv')
    OUTPUT = os.path.join(BASE_DIR, 'titanic_preprocessing')
    run_preprocessing(INPUT, OUTPUT)