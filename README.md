# 📘 Judul Proyek

**Klasifikasi Kategori Produk E-Commerce Berbasis Analisis Clickstream Menggunakan Deep Learning**

## 👤 Informasi

- **Nama:** Muhammad Hasanuddin
- **Repo:** https://github.com/Isan955/DS-UAS2025
- **Video:** [...]

---

# 1. 🎯 Ringkasan Proyek

- Menyelesaikan permasalahan klasifikasi multi-kelas pada data _clickstream_ e-commerce untuk memprediksi kategori produk utama yang diminati pengguna.
- Melakukan _data preparation_ yang ketat (Standardization dan Label Encoding).
- Membangun 3 model: **Logistic Regression (Baseline)**, **Random Forest (Advanced)**, dan **Multilayer Perceptron (Deep Learning)**.
- Melakukan evaluasi menggunakan **Accuracy** dan **Weighted F1-Score** untuk menentukan model terbaik.

---

# 2. 📄 Problem & Goals

**Problem Statements:**

- Perusahaan kesulitan mengidentifikasi secara cepat dan akurat kategori produk (`page 1`) yang sedang dilihat pengguna berdasarkan aktivitas _clickstream_ mereka.
- Model linear tradisional tidak mampu menangkap hubungan kompleks (non-linear) antar fitur seperti urutan klik (`order`) dan harga (`price`).

**Goals:**

- Membangun model klasifikasi yang mampu memprediksi kategori produk (`Trousers`, `Skirts`, `Blouses`, `Sale`) dengan akurasi setinggi mungkin.
- Menentukan apakah model _Machine Learning_ tingkat lanjut (Random Forest) atau _Deep Learning_ (MLP) memberikan keunggulan performa pada data tabular non-linear.

---

## 📁 Struktur Folder

```
project/
│
├── data/
│   ├── e_sho_clothing_cleaned.csv
│   └── e_sho_clothing.csv
│
├── notebooks/
│   └── 234311045_Muhammad Hasanuddin_UAS_DS.ipynb
│
├── src/
│
├── models/
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   └── model_deep_learning.h5
│
├── images/
│   ├── Deep_Learning.png
│   ├── DeepLearningAccuracy.png
│   ├── DeepLearningLoss.png
│   ├── Distribusi_harga_perkategori.png
│   ├── Heatmap_korelasi_fitur.png
│   ├── LogReg.png
│   ├── model_comparison_with_f1-score.png
│   ├── Model_Comparison.png
│   ├── Random_Forest.png
│   └── Target.png
│
├── Laporan Proyek Machine Learning.md
├── Cheklist Submit.md
├── License
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 3. 📊 Dataset

- **Sumber:** https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
- **Jumlah Data:** 165,474 samples (sebelum _data cleaning_)
- **Tipe:** Tabular (Clickstream Data)

### Fitur Utama

| Fitur                     | Deskripsi                                                        |
| ------------------------- | ---------------------------------------------------------------- |
| `order`                   | Urutan klik dalam sesi (1 hingga 200)                            |
| `page 2 (clothing model)` | Kode ID unik untuk model pakaian (217 tipe)                      |
| `colour`                  | Kode ID warna (14 tipe)                                          |
| `price`                   | Harga produk yang dilihat                                        |
| `page 1 (main category)`  | Target: Kategori utama (1=Trousers, 2=Skirts, 3=Blouses, 4=Sale) |

---

# 4. 🔧 Data Preparation

- **Cleaning:**
  - Tidak ada _Missing Values_.
  - Menghapus baris **duplikat sempurna**.
  - Menghapus fitur tidak relevan (`year`, `session ID`).
- **Transformasi:**
  - **StandardScaler** diterapkan pada seluruh fitur input (`X`) untuk menstandarisasi data dan mengurangi dampak _outliers_ pada model linear dan DL.
  - **Label Encoding** diterapkan pada variabel target (`y`) untuk format yang dibutuhkan oleh Keras (`sparse_categorical_crossentropy`).
- **Splitting:**
  - Train set: 80% (132.379)
  - Test set: 20% (33.095)
  - Menggunakan **Stratified Split** untuk mempertahankan distribusi kelas yang relatif seimbang.

---

# 5. 🤖 Modeling

- **Model 1 – Baseline:** **Logistic Regression** (max_iter=1000).
- **Model 2 – Advanced ML:** **Random Forest Classifier** (n_estimators=100).
- **Model 3 – Deep Learning:** **Multilayer Perceptron (MLP)**. Arsitektur: Input Dense(64, ReLU) Dropout(0.3) Dense(32, ReLU) Dropout(0.2) Dense(4, Softmax).

---

# 6. 🧪 Evaluation

**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)

### Hasil Singkat

| Model         | Score (Accuracy) | Catatan                                                    |
| ------------- | ---------------- | ---------------------------------------------------------- |
| Baseline      | 65.21%           | Model linear gagal menangkap pola non-linear.              |
| Advanced      | 98.75%           | Peningkatan dramatis, membuktikan kompleksitas non-linear. |
| Deep Learning | 98.67%           | Model, Weighted F1-Score **0.99**.                         |

---

# 7. 🏁 Kesimpulan

- Model terbaik: **Random Forest Classifier**.
- Alasan: Random Forest mencapai Akurasi tertinggi (0.9875) dan F1-Score yang identik (0.9875) pada test set. Meskipun model Deep Learning (MLP) juga memberikan performa sangat tinggi (0.9867), Random Forest memberikan hasil yang sedikit lebih unggul dan efisien dalam hal komputasi dan interpretasi dibandingkan arsitektur MLP yang lebih kompleks. Kinerja superior ini membuktikan keunggulan model ensemble dalam menangkap hubungan non-linear antar fitur perilaku pengguna.
- Insight penting: Perbedaan signifikan antara performa LogReg (65.21%) dan model non-linear (RF/MLP) membuktikan bahwa data _clickstream_ memiliki hubungan **sangat non-linear** yang hanya dapat ditangkap oleh model _ensemble_ atau _neural network_.

---

# 8. 🔮 Future Work

- [ ] Tuning model Random Forest (RF) untuk mencoba menyamai/melampaui performa MLP, karena RF lebih mudah di-_deploy_ dalam skenario _real-time_.
- [ ] Analisis _Feature Importance_ dari Random Forest untuk mendapatkan wawasan bisnis yang lebih jelas.
- [ ] Deployment model terbaik ke sistem rekomendasi e-commerce.

---

# 9. 🔁 Reproducibility

Gunakan environment:
