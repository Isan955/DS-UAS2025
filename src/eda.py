# 3. EXPLORATORY DATA ANALYSIS (EDA)
target_col = 'page 1 (main category)'
price_col = 'price'

# Visualisasi 1: Distribusi Kelas
plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x=target_col, palette='viridis')
plt.title('Distribusi Kategori Produk (Target)')
plt.xlabel('Kategori (1=Trousers, 2=Skirts, 3=Blouses, 4=Sale)')
plt.show()

# Visualisasi 2: Hubungan Harga vs Kategori
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x=target_col, y=price_col, palette='Set2')
plt.title('Distribusi Harga per Kategori')
plt.ylim(0, 100) # Zoom in harga 0-100 agar terlihat jelas
plt.show()

# Visualisasi 3: Korelasi Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df_clean.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur')
plt.show()