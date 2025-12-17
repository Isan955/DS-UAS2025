# EVALUATION & COMPARISON
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from IPython.display import display
import os

models_dict = {
    'Logistic Regression (Baseline)': model_lr,
    'Random Forest (Advanced)': model_rf,
    'Deep Learning (MLP)': model_dl
}

# Container hasil
final_metrics = []

print("HASIL EVALUASI PADA DATA TEST ")

# Loop untuk mengevaluasi setiap model
for name, model in models_dict.items():
    print(f"\n>> Mengevaluasi Model: {name}...")

    # 2. Lakukan Prediksi
    if name == 'Deep Learning (MLP)':
        y_pred_prob = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        # Prediksi SKLEARN (LogReg, RF)
        y_pred = model.predict(X_test_scaled)

    # 3. Hitung Metrik
    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    final_metrics.append({
        'Model': name,
        'Accuracy': acc,
        'F1-Score': f1
    })

    # Target names (0:Trousers, 1:Skirts, 2:Blouses, 3:Sale)
    target_names = ['Trousers', 'Skirts', 'Blouses', 'Sale']
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# 4. Membuat Tabel DataFrame
df_metrics = pd.DataFrame(final_metrics)

print("\n=== TABEL PERBANDINGAN AKHIR ===")
# Format tampilan di Jupyter/Colab
display(df_metrics.style.format({
    'Accuracy': '{:.4f}',
    'F1-Score': '{:.4f}'
}))

# 5. Visualisasi Perbandingan Metrik
plt.figure(figsize=(10, 6))


df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Plot Bar Chart
ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Perbandingan Performa Model (Accuracy vs Weighted F1)", fontsize=14)
plt.ylim(0, 1.05) # Batas y hingga 1.05
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

# Label Angka di atas batang
for container in ax.containers:

    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig('images/viz_model_comparison_metrics.png')
plt.show()