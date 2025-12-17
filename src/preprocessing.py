# 4. PREPROCESSING (SPLIT & SCALE)
target_col = 'page 1 (main category)'

# 'page 2 (clothing model)' memberi tahu jawaban secara langsung.
leak_cols = ['page 2 (clothing model)']

# Pisahkan X (Fitur) dan y (Target)
X = df_clean.drop(columns=[target_col] + leak_cols, errors='ignore')
y = df_clean[target_col]

# Ubah target menjadi 0,1,2,3
y = y - 1

print(f"Fitur yang digunakan (X): {list(X.columns)}")

# 4. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan Scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Gabungkan X dan y untuk disimpan
processed_train = pd.DataFrame(X_train_scaled, columns=X.columns)
processed_train['target'] = y_train.values
processed_train.to_csv('data/processed/train_data_processed.csv', index=False)

processed_test = pd.DataFrame(X_test_scaled, columns=X.columns)
processed_test['target'] = y_test.values
processed_test.to_csv('data/processed/test_data_processed.csv', index=False)

print("✅ Data Train/Test yang sudah di-scale tersimpan di 'data/processed/'")