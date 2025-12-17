# 2. DATA LOADING & CLEANING
df = pd.read_csv('/content/e-shop clothing 2008.csv', delimiter=';')

print(f"Data awal: {df.shape}")

# Hapus kolom tidak penting
cols_to_drop = ['year', 'session ID']
df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Hapus duplikat & missing values
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna()

# TAMBAHAN PENTING: UBAH TEKS KE ANGKA (FIX ERROR 'C12')
from sklearn.preprocessing import LabelEncoder

# Cari kolom yang tipe datanya 'object' (teks/string)
object_cols = df_clean.select_dtypes(include=['object']).columns

# loop untuk mengubah teks menjadi angka
if len(object_cols) > 0:
    print(f"Terdeteksi kolom teks: {list(object_cols)}. Melakukan Encoding...")
    le = LabelEncoder()
    for col in object_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        print(f"   -> Kolom '{col}' berhasil diubah menjadi angka.")
else:
    print("Semua kolom sudah berupa angka.")

print(f"Data setelah cleaning: {df_clean.shape}")

df_clean.to_csv('data/processed/e_shop_clothing_final_cleaned.csv', index=False)
print("Data bersih tersimpan di 'data/processed/e_shop_clothing_final_cleaned.csv'")