# 8. EVALUATION & VISUALIZATION
print("\n 8. FINAL EVALUATION")

# 8.1 Plot Training History (Deep Learning)
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Deep Learning Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Deep Learning Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 8.2 Confusion Matrix Comparison
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plot_cm(y_test, y_pred_lr, f'LogReg (Acc: {acc_lr:.2%})')

plt.subplot(1, 3, 2)
plot_cm(y_test, y_pred_rf, f'Random Forest (Acc: {acc_rf:.2%})')

plt.subplot(1, 3, 3)
plot_cm(y_test, y_pred_dl, f'Deep Learning (Acc: {acc_dl:.2%})')

plt.tight_layout()
plt.show()

# 8.3 Final Classification Report (Best Model - MLP)
print(classification_report(y_test, y_pred_dl, target_names=['Trousers', 'Skirts', 'Blouses', 'Sale']))