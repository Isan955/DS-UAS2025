np.random.seed(42)
tf.random.set_seed(42)

# Buat folder untuk menyimpan output
for folder in ['models', 'data/processed', 'images']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' berhasil dibuat.")

print("Libraries berhasil di-import dan struktur folder siap.")