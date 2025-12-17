# 5. MODELING 1: LOGISTIC REGRESSION (Baseline)
print("\n 5. TRAINING MODEL 1: LOGISTIC REGRESSION")

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train)

y_pred_lr = model_lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f" Akurasi Logistic Regression: {acc_lr:.4f}")

# Simpan Model
joblib.dump(model_lr, 'models/model_logistic_regression.pkl')

# 6. MODELING 2: RANDOM FOREST (Advanced ML)
print("\n 6. TRAINING MODEL 2: RANDOM FOREST")

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Random Forest tidak wajib pakai data scaled, tapi tidak masalah jika pakai
model_rf.fit(X_train_scaled, y_train)

y_pred_rf = model_rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f" Akurasi Random Forest: {acc_rf:.4f}")

# Simpan Model
joblib.dump(model_rf, 'models/model_random_forest.pkl')

# Feature Importance
importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop 3 Fitur Paling Penting (Random Forest):")
for f in range(3):
    print(f"{f+1}. {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")


# 7. MODELING 3: DEEP LEARNING (MLP)
print("\7. TRAINING MODEL 3: DEEP LEARNING (MLP)")

input_dim = X_train_scaled.shape[1]
num_classes = 4 # (0, 1, 2, 3)

# Arsitektur Model
model_dl = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),                 # Regularisasi
    Dense(32, activation='relu'),
    Dropout(0.2),                 # Regularisasi
    Dense(num_classes, activation='softmax') # Output Layer
])

# Compile
model_dl.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model_dl.fit(
    X_train_scaled, y_train,
    validation_split=0.2, # 20% dari train dipakai validasi
    epochs=15,            # Bisa ditambah jika belum konvergen
    batch_size=64,
    verbose=1
)

# Evaluasi
loss_dl, acc_dl = model_dl.evaluate(X_test_scaled, y_test, verbose=0)
print(f" Akurasi Deep Learning: {acc_dl:.4f}")

# Prediksi kelas
y_pred_dl_proba = model_dl.predict(X_test_scaled)
y_pred_dl = np.argmax(y_pred_dl_proba, axis=1)

# Simpan Model
model_dl.save('models/model_deep_learning.h5')
print(" Semua model berhasil disimpan di folder 'models/'")

print(model_dl.summary())