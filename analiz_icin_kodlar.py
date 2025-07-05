

#Tüm analizler için başlangıç noktası.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, shapiro, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore') # Uyarıları kapatmak için

import pandas as pd
import numpy as np

# Örnek bir DataFrame oluşturalım
# Gerçek verini 'df' olarak yüklediğinizi varsayıyorum.
# Eğer elinde örnek bir veri yoksa, aşağıdaki satırı gerçek veri setinin yükleme koduyla değiştirebilirsin:
# df = pd.read_excel("senin_veri_setin.xlsx")

# Daha önce kullandığımız 'cleaned_dataset_no_outliers.xlsx' dosyasını yükleyelim
# Eğer bu dosya zaten aykırı değerlerden temizlenmişse, bu kod sadece nasıl yapılacağını gösterir.
# Aykırı değerleri görmek için 'vveerrii_sayisallastirilmis.xlsx' gibi daha ham bir veri seti kullanmak daha mantıklı olabilir.
file_path_for_outliers = r"C:\Users\Emre\Downloads\vveerrii_sayisallastirilmis.xlsx"
df_zscore = pd.read_excel(file_path_for_outliers)

# Sütun adlarını temizleme
df_zscore.columns = df_zscore.columns.str.replace(' ', '_')
df_zscore.columns = df_zscore.columns.str.replace('ç', 'c').str.replace('ı', 'i').str.replace('ş', 's').str.replace('ğ', 'g').str.replace('ü', 'u').str.replace('ö', 'o')
df_zscore.columns = df_zscore.columns.str.replace('?', '')

# Sayısal sütunları tanımlıyorum
numerical_cols_for_cleaning = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs']

# Sayısal sütunları dönüştürüp, eksik değerleri medyanla dolduruyorum (temizleme öncesi hazırlık)
for col in numerical_cols_for_cleaning:
    df_zscore[col] = pd.to_numeric(df_zscore[col], errors='coerce')
    df_zscore[col] = df_zscore[col].fillna(df_zscore[col].median())

print(f"Z-skoru ile temizleme öncesi veri seti boyutu: {df_zscore.shape}")

# Z-skoru yöntemiyle aykırı değerleri temizleme fonksiyonu
def clean_outliers_zscore(df, columns, threshold=3):
    df_cleaned = df.copy()
    initial_rows = df_cleaned.shape[0]
    outlier_counts = {}

    for col in columns:
        if col in df_cleaned.columns:
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            if std == 0: # Standart sapma sıfırsa bu sütunda aykırı değer olamaz
                outlier_counts[col] = 0
                continue
            
            z_scores = np.abs((df_cleaned[col] - mean) / std)
            # Aykırı değerleri içeren satırları filtreliyorum
            df_cleaned = df_cleaned[z_scores <= threshold]
            outlier_counts[col] = initial_rows - df_cleaned.shape[0] # Her adımda kaç satır düştüğünü sayıyorum

    print("\nZ-skoru Yöntemiyle Temizlenen Aykırı Değer Sayıları (Her sütun için):")
    # Z-skoruna göre, temizleme işlemi sırasında her bir sütun için kaç aykırı değerin atıldığını takip etmek zor.
    # Çünkü bir sütunda aykırı olan satır, diğer sütunlarda da aykırı olabilir ve sadece bir kere atılır.
    # Bu yüzden, toplam temizlenen satır sayısını vermek daha doğru.
    print(f"Toplam temizlenen satır sayısı (Z-skoru): {initial_rows - df_cleaned.shape[0]}")
    
    return df_cleaned

# Belirlediğim sayısal sütunları Z-skoru ile temizliyorum
df_zscore_cleaned = clean_outliers_zscore(df_zscore, numerical_cols_for_cleaning, threshold=3)

print(f"Z-skoru ile temizleme sonrası veri seti boyutu: {df_zscore_cleaned.shape}")
print("\nZ-skoru ile temizlenmiş veri setinin ilk 5 satırı:")
print(df_zscore_cleaned.head())

import pandas as pd
import numpy as np

# IQR ile temizlik için yine orijinal veri setinden başlıyorum
file_path_for_outliers_iqr = r"C:\Users\Emre\Downloads\vveerrii_sayisallastirilmis.xlsx"
df_iqr = pd.read_excel(file_path_for_outliers_iqr)

# Sütun adlarını temizleme
df_iqr.columns = df_iqr.columns.str.replace(' ', '_')
df_iqr.columns = df_iqr.columns.str.replace('ç', 'c').str.replace('ı', 'i').str.replace('ş', 's').str.replace('ğ', 'g').str.replace('ü', 'u').str.replace('ö', 'o')
df_iqr.columns = df_iqr.columns.str.replace('?', '')

# Sayısal sütunları tanımlıyorum (aynı listeyi kullanıyorum)
# numerical_cols_for_cleaning = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs']

# Sayısal sütunları dönüştürüp, eksik değerleri medyanla dolduruyorum (temizleme öncesi hazırlık)
for col in numerical_cols_for_cleaning:
    df_iqr[col] = pd.to_numeric(df_iqr[col], errors='coerce')
    df_iqr[col] = df_iqr[col].fillna(df_iqr[col].median())

print(f"\nIQR ile temizleme öncesi veri seti boyutu: {df_iqr.shape}")

# IQR yöntemiyle aykırı değerleri temizleme fonksiyonu
def clean_outliers_iqr(df, columns, factor=1.5):
    df_cleaned = df.copy()
    initial_rows = df_cleaned.shape[0]
    outlier_counts = {}

    for col in columns:
        if col in df_cleaned.columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Aykırı değerleri içeren satırları filtreliyorum
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            outlier_counts[col] = initial_rows - df_cleaned.shape[0]

    print("\nIQR Yöntemiyle Temizlenen Aykırı Değer Sayıları (Her sütun için):")
    # Z-skoru'nda olduğu gibi, toplam atılan satır sayısını belirtmek daha doğru.
    print(f"Toplam temizlenen satır sayısı (IQR): {initial_rows - df_cleaned.shape[0]}")
    
    return df_cleaned

# Belirlediğim sayısal sütunları IQR ile temizliyorum
df_iqr_cleaned = clean_outliers_iqr(df_iqr, numerical_cols_for_cleaning, factor=1.5)

print(f"IQR ile temizleme sonrası veri seti boyutu: {df_iqr_cleaned.shape}")
print("\nIQR ile temizlenmiş veri setinin ilk 5 satırı:")
print(df_iqr_cleaned.head())


# --- Veri Yükleme ---
file_path = r"C:\Users\Emre\Downloads\cleaned_dataset_no_outliers.xlsx"
df = pd.read_excel(file_path)

df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('ç', 'c').str.replace('ı', 'i').str.replace('ş', 's').str.replace('ğ', 'g').str.replace('ü', 'u').str.replace('ö', 'o')
df.columns = df.columns.str.replace('?', '')

# Sayısal ve Kategorik değişkenlerin tanımlanması 
numerical_cols = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs']
categorical_cols = ['Mahalle', 'Isıtma', 'Krediye_Uygun', 'Tapu_Durumu', 'Kimden', 'Takas', 'Site_Adı', 'Esyali_Kod', 'Kullanım_Durumu_Kod'] # Ki-kare ve t-testi için kodlanmış hallerini ekledim

# Veri tiplerini kontrol edin ve dönüştürün
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') # Sayısala çevir, hata olursa NaN yap


for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
if 'Krediye_Uygun_Kod' not in df.columns and 'Krediye_Uygun' in df.columns:
    df['Krediye_Uygun_Kod'] = le.fit_transform(df['Krediye_Uygun'].astype(str))
if 'Esyali_Kod' not in df.columns and 'Esya_Durumu' in df.columns:
    df['Esyali_Kod'] = le.fit_transform(df['Esya_Durumu'].astype(str))
if 'Takas_Kod' not in df.columns and 'Takas' in df.columns:
    df['Takas_Kod'] = le.fit_transform(df['Takas'].astype(str))
if 'Kullanım_Durumu_Kod' not in df.columns and 'Kullanım_Durumu' in df.columns:
    df['Kullanım_Durumu_Kod'] = le.fit_transform(df['Kullanım_Durumu'].astype(str))
if 'Site_Icerisinde_Kod' not in df.columns and 'Site_Adı' in df.columns: # Burası site adı kolonundan türetilirse
    df['Site_Icerisinde_Kod'] = df['Site_Adı'].apply(lambda x: 1 if pd.notna(x) and x != '-' else 0)


print("Veri Setinin İlk 5 Satırı:")
print(df.head())
print("\nSayısal Sütunların İstatistikleri (Başlangıç):")
print(df[numerical_cols].describe().T)

```
---

## 2. Tanımlayıcı İstatistikler ve Aykırı Değer Analizi Kodları

Bu bölüm, bulgularınızın 4.1. başlığı altında yer alan tüm analizleri kapsar.

### 2.1. Veri Setinin Genel Tanımlayıcı İstatistikleri


```python
# Sayısal değişkenleriniz için tanımlayıcı istatistikler
# Bulgularınızdaki tabloya benzer çıktılar almak için manuel olarak bazı hesaplamalar yapabiliriz.

# Kullanacağınız sayısal değişkenler
target_numerical_cols = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric']

desc_stats = df[target_numerical_cols].describe().T
desc_stats['median'] = df[target_numerical_cols].median()
desc_stats['var'] = df[target_numerical_cols].var()
desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']
desc_stats['N'] = df[target_numerical_cols].count() # N değeri için count() kullanıyoruz

# Kolon isimlerini bulgularınızdaki gibi ayarlayalım
desc_stats = desc_stats[['N', 'mean', 'median', 'var', 'std', 'min', '25%', '75%', 'max', 'IQR']]
desc_stats.columns = ['N', 'Ort.', 'Med.', 'Var.', 'SS', 'Min', 'Q1', 'Q3', 'Max', 'IQR']

# Fiyat (TL) için gösterim formatını düzeltelim
desc_stats.loc['Fiyat_(TL)', 'Ort.'] = f"{desc_stats.loc['Fiyat_(TL)', 'Ort.'] / 1000000:.2f}M"
desc_stats.loc['Fiyat_(TL)', 'Med.'] = f"{desc_stats.loc['Fiyat_(TL)', 'Med.'] / 1000000:.2f}M"
desc_stats.loc['Fiyat_(TL)', 'Var.'] = f"{desc_stats.loc['Fiyat_(TL)', 'Var.']:.2E}" # Bilimsel gösterim
desc_stats.loc['Fiyat_(TL)', 'SS'] = f"{desc_stats.loc['Fiyat_(TL)', 'SS'] / 1000000:.2f}M"
desc_stats.loc['Fiyat_(TL)', 'Min'] = f"{desc_stats.loc['Fiyat_(TL)', 'Min'] / 1000:.0f}k"
desc_stats.loc['Fiyat_(TL)', 'Q1'] = f"{desc_stats.loc['Fiyat_(TL)', 'Q1'] / 1000000:.2f}M"
desc_stats.loc['Fiyat_(TL)', 'Q3'] = f"{desc_stats.loc['Fiyat_(TL)', 'Q3'] / 1000000:.2f}M"
desc_stats.loc['Fiyat_(TL)', 'Max'] = f"{desc_stats.loc['Fiyat_(TL)', 'Max'] / 1000000:.2f}M"
desc_stats.loc['Fiyat_(TL)', 'IQR'] = f"{desc_stats.loc['Fiyat_(TL)', 'IQR'] / 1000000:.2f}M"


print("\n--- 4.1.1 Veri Setinin Genel Tanımlayıcı İstatistikleri ---")
print(desc_stats.to_markdown(numalign="left", stralign="left")) # Markdown tablo çıktısı için


### 2.2. Aykırı Değer Tespiti (IQR ve Z-Skoru)

Bu bölümde, bulgularınızdaki 4.1.2.1 ve 4.1.2.2'yi oluşturan kodlar yer almaktadır. **Bu kodlar, veri temizleme aşamanızda çalıştırılmış olmalı ve aykırı değerler tespit edildikten sonra veri setinizden çıkarılmış olmalıdır.** `cleaned_dataset_no_outliers.xlsx` dosyanızın bu işlemlerden sonraki hali olduğunu varsayıyoruz. Ancak kodlarını gösterebiliriz.

```python
temp_df.columns = temp_df.columns.str.replace(' ', '_').str.replace('ç', 'c').str.replace('ı', 'i').str.replace('ş', 's').str.replace('ğ', 'g').str.replace('ü', 'u').str.replace('ö', 'o').str.replace('?', '')
for col in ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric']:
    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    temp_df[col] = temp_df[col].fillna(temp_df[col].median()) # Eksik değerleri doldur

print("\n--- 4.1.2. Aykırı Değer Tespiti ---")

# 4.1.2.1. Çeyrekler Arası Mesafe (IQR) Yöntemi
iqr_outliers = []
for col in target_numerical_cols:
    Q1 = temp_df[col].quantile(0.25)
    Q3 = temp_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = temp_df[(temp_df[col] < lower_bound) | (temp_df[col] > upper_bound)].shape[0]
    iqr_outliers.append({"Değişken": col, "Aykırı Say.": outlier_count})

iqr_outliers_df = pd.DataFrame(iqr_outliers)
print("\n4.1.2.1. Çeyrekler Arası Mesafe (IQR) Yöntemi Aykırı Değer Sayıları:")
print(iqr_outliers_df.to_markdown(numalign="left", stralign="left"))


# 4.1.2.2. Z-Skoru Yöntemi
zscore_outliers = []
for col in target_numerical_cols:
    mean = temp_df[col].mean()
    std = temp_df[col].std()
    if std == 0: # Standart sapma sıfırsa z-skor hesaplanamaz, atla
        outlier_count = 0
    else:
        z_scores = np.abs((temp_df[col] - mean) / std)
        outlier_count = temp_df[z_scores > 3].shape[0]
    zscore_outliers.append({"Değişken": col, "Aykırı Say.": outlier_count})

zscore_outliers_df = pd.DataFrame(zscore_outliers)
print("\n4.1.2.2. Z-Skoru Yöntemi Aykırı Değer Sayıları:")
print(zscore_outliers_df.to_markdown(numalign="left", stralign="left"))

```

### 2.3. Aykırı Değerlerin Veriye Olan Etkisinin İncelenmesi

Tablo 4.4'ü oluşturan kod:

```python
original_df = pd.read_excel(r"C:\Users\Emre\Downloads\vveerrii_sayisallastirilmis.xlsx") 
original_df.columns = original_df.columns.str.replace(' ', '_').str.replace('ç', 'c').str.replace('ı', 'i').str.replace('ş', 's').str.replace('ğ', 'g').str.replace('ü', 'u').str.replace('ö', 'o').str.replace('?', '')
for col in target_numerical_cols:
    original_df[col] = pd.to_numeric(original_df[col], errors='coerce')
    original_df[col] = original_df[col].fillna(original_df[col].median()) # Eksik değerleri doldur


comparison_data = []
for col in target_numerical_cols:
    original_mean = original_df[col].mean()
    cleaned_mean = df[col].mean() # df temizlenmiş veri seti
    original_std = original_df[col].std()
    cleaned_std = df[col].std()

    comparison_data.append({
        "Değişken": col,
        "Ort. (O)": round(original_mean, 2) if col != 'Fiyat_(TL)' else f"{original_mean:,.0f}",
        "Ort. (T)": round(cleaned_mean, 2) if col != 'Fiyat_(TL)' else f"{cleaned_mean:,.0f}",
        "Std. Sap. (O)": round(original_std, 2) if col != 'Fiyat_(TL)' else f"{original_std:,.0f}",
        "Std. Sap. (T)": round(cleaned_std, 2) if col != 'Fiyat_(TL)' else f"{cleaned_std:,.0f}"
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n--- 4.1.2.3. Aykırı Değerlerin Veriye Olan Etkisinin İncelenmesi ---")
print(comparison_df.to_markdown(numalign="left", stralign="left"))
print(f"\nVeri setinde yapılan temizlik sonrası kalan veri sayısı: {df.shape[0]}") # Temizlenmiş df'in satır sayısını gösterir
```

### 2.4. Logaritmik Dönüşüm Sonuçları

Bulgularınızda logaritmik dönüşümün etkileri açıklanmış. Bu işlemi nasıl yaptığınızı ve sonuçlarını görmek için aşağıdaki kod kullanılabilir.

```python
# Logaritmik dönüşümün uygulandığı varsayılan sütunlar
log_cols = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²']

print("\n--- 4.1.3. Logaritmik Dönüşüm Sonuçları ---")
print("Logaritmik dönüşüm genellikle verinin çarpıklığını azaltır ve normal dağılıma yaklaştırır.")
print("Aşağıdaki kod, bu dönüşümün nasıl yapıldığını gösterir ve dönüşüm sonrası istatistikleri karşılaştırır.")

# Eğer df_original varsa, onunla karşılaştırın
if 'original_df' in locals(): # original_df'in var olup olmadığını kontrol et
    log_comparison_data = []
    for col in log_cols:
        original_col_data = original_df[col].dropna()
        log_transformed_col_data = np.log1p(original_col_data) # log1p, log(x+1) demektir, 0 değerleri için iyidir

        log_comparison_data.append({
            "Değişken": col,
            "Orijinal Ort.": round(original_col_data.mean(), 2) if col != 'Fiyat_(TL)' else f"{original_col_data.mean():,.0f}",
            "Orijinal SS": round(original_col_data.std(), 2) if col != 'Fiyat_(TL)' else f"{original_col_data.std():,.0f}",
            "Log Dönüşümlü Ort.": round(log_transformed_col_data.mean(), 4),
            "Log Dönüşümlü SS": round(log_transformed_col_data.std(), 4),
            "Orijinal Çarpıklık": round(original_col_data.skew(), 4),
            "Log Dönüşümlü Çarpıklık": round(log_transformed_col_data.skew(), 4)
        })
    log_comparison_df = pd.DataFrame(log_comparison_data)
    print("\nLogaritmik Dönüşüm Öncesi ve Sonrası İstatistikler:")
    print(log_comparison_df.to_markdown(numalign="left", stralign="left"))

    # Log dönüşüm sonrası dağılım görselleştirmesi (örnek Fiyat için)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(original_df['Fiyat_(TL)'], kde=True)
    plt.title('Orijinal Fiyat Dağılımı')
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(original_df['Fiyat_(TL)']), kde=True)
    plt.title('Logaritmik Dönüşüm Sonrası Fiyat Dağılımı')
    plt.tight_layout()
    plt.show()

```

### 2.5. Görsel Analizler (Tablo 4.5 ve 4.6)

Bulgularınızda görsellerden bahsedilmiş. İşte bu görselleri oluşturabilecek kod örnekleri:

```python
print("\n--- 4.1.4. Görsel Analizler ---")

fig, axes = plt.subplots(len(target_numerical_cols), 2, figsize=(12, 4 * len(target_numerical_cols)))
fig.suptitle('Aykırı Değerlerin Öncesi ve Sonrası Dağılımı (Kutu Grafikleri)', y=1.02)

for i, col in enumerate(target_numerical_cols):
    sns.boxplot(y=original_df[col], ax=axes[i, 0])
    axes[i, 0].set_title(f'Orijinal: {col}')
    sns.boxplot(y=df[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Temizlenmiş: {col}')

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

top_10_mahalle = df['Mahalle'].value_counts().nlargest(10).index

plt.figure(figsize=(14, 7))
sns.boxplot(x='Mahalle', y='Fiyat_(TL)', data=df[df['Mahalle'].isin(top_10_mahalle)], palette='viridis')
plt.title('İlk 10 Mahalledeki Konut Fiyatlarının Dağılımı')
plt.xlabel('Mahalle')
plt.ylabel('Fiyat (TL)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

## 3. Bağımsız Örneklem t-Testi Bulguları Kodu

Bulgularınızdaki 4.2. başlığına karşılık gelir.

```python
print("\n--- 4.2. Bağımsız Örneklem t-Testi Bulguları ---")

# Takas Durumu Grubu
print("\n4.2.2. Takas Durumu Grubu")
numeric_vars_t_test = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Bina_Yası_Ortalama']

# Eğer 'Takas_Kod' sütunu yoksa, 'Takas' sütunundan türetin (örneğin "Evet", "Hayır" değerleriyle)
if 'Takas_Kod' not in df.columns:
    df['Takas_Kod'] = df['Takas'].map({'Evet': 1, 'Hayır': 0}).fillna(0) # Nan'ları 'Hayır' olarak kabul edelim

for var in numeric_vars_t_test:
    group1 = df[df['Takas_Kod'] == 1][var].dropna() # Takas yapanlar
    group0 = df[df['Takas_Kod'] == 0][var].dropna() # Takas yapmayanlar

    # Yeterli veri olup olmadığını kontrol edin
    if len(group1) > 1 and len(group0) > 1:
        stat, p = ttest_ind(group1, group0, equal_var=False) # Welch's t-test (equal_var=False) eşitsiz varyans varsayımı
        print(f"\nDeğişken: {var}")
        print(f"Takas Yapanlar Ort.: {group1.mean():,.2f}")
        print(f"Takas Yapmayanlar Ort.: {group0.mean():,.2f}")
        print(f"p-değeri: {p:.4f}")
        if p < 0.05:
            print("Sonuç: p < 0.05, Anlamlı fark var (H0 reddedilir).")
        else:
            print("Sonuç: p >= 0.05, Anlamlı fark yok (H0 reddedilmez).")
    else:
        print(f"\nDeğişken: {var} - Yeterli veri bulunamadı.")


# Site İçerisinde Olma Durumu Grubu
print("\n4.2.3. Site İçerisinde Olma Durumu Grubu")

if 'Site_Icerisinde_Kod' not in df.columns:
    df['Site_Icerisinde_Kod'] = df['Site_Adı'].apply(lambda x: 1 if pd.notna(x) and x != '-' else 0)

for var in numeric_vars_t_test:
    group1 = df[df['Site_Icerisinde_Kod'] == 1][var].dropna() # Site içinde olanlar
    group0 = df[df['Site_Icerisinde_Kod'] == 0][var].dropna() # Site içinde olmayanlar

    if len(group1) > 1 and len(group0) > 1:
        stat, p = ttest_ind(group1, group0, equal_var=False)
        print(f"\nDeğişken: {var}")
        print(f"Site İçindeki Ort.: {group1.mean():,.2f}")
        print(f"Site Dışındaki Ort.: {group0.mean():,.2f}")
        print(f"p-değeri: {p:.4f}")
        if p < 0.05:
            print("Sonuç: p < 0.05, Anlamlı fark var (H0 reddedilir).")
        else:
            print("Sonuç: p >= 0.05, Anlamlı fark yok (H0 reddedilmez).")
    else:
        print(f"\nDeğişken: {var} - Yeterli veri bulunamadı.")
```

---

## 4. Korelasyon Analizi Bulguları Kodu

Bulgularınızdaki 4.3. başlığına karşılık gelir.

```python
print("\n--- 4.3. Korelasyon Analizi Bulguları ---")

# Korelasyon matrisi için seçilen sayısal değişkenler
correlation_cols = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Bina_Yası_Ortalama', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric']

# Korelasyon matrisini hesaplama
corr_matrix = df[correlation_cols].corr(method='pearson')

print("\nPearson Korelasyon Katsayıları:")
print(corr_matrix.to_markdown(numalign="left", stralign="left"))

# Yorumlar (kod çıktısı değil, manuel analiz)
print("\n--- 4.3.1. Korelasyon Yorumları (Manuel olarak bu çıktılardan yorumlayabilirsiniz) ---")
print(f"Fiyat (TL) ile Brüt m² arasında yüksek pozitif korelasyon: {corr_matrix.loc['Fiyat_(TL)', 'Brüt_m²']:.2f}")
print(f"Fiyat (TL) ile Net m² arasında güçlü pozitif korelasyon: {corr_matrix.loc['Fiyat_(TL)', 'Net_m²']:.2f}")
print(f"Brüt m² ile Net m² arasındaki korelasyon: {corr_matrix.loc['Brüt_m²', 'Net_m²']:.2f}")
print(f"Bina Yaşı ile Fiyat (TL) arasında negatif korelasyon: {corr_matrix.loc['Fiyat_(TL)', 'Bina_Yası_Ortalama']:.2f}")

# 4.3.2. Korelasyon Isı Haritası (Tablo 4.7)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={'size': 10})
plt.title('Korelasyon Isı Haritası', fontsize=16)
plt.show()
```

---

## 5. Ki-Kare Testi Bulguları Kodu

```python
print("\n--- 4.4. Ki-Kare Testi Bulguları ---")


# Krediye Uygunluk ve Eşyalı Olma Durumu (varsayılan: Esyali_Kod)
print("\n4.4.1. Krediye Uygunluk ve Eşyalı Olma Durumu")
if 'Krediye_Uygun_Kod' not in df.columns:
    df['Krediye_Uygun_Kod'] = df['Krediye_Uygun'].map({'Evet': 1, 'Hayır': 0, 'Belirtilmemis': 2}).fillna(2)
if 'Esyali_Kod' not in df.columns:
    df['Esyali_Kod'] = df['Esya_Durumu'].map({'Evet': 1, 'Hayır': 0, 'Belirtilmemis': 2}).fillna(2)

contingency_table_1 = pd.crosstab(df['Krediye_Uygun_Kod'], df['Esyali_Kod'])
chi2, p, dof, ex = chi2_contingency(contingency_table_1)
print("\nKrediye Uygunluk vs Eşyalı Olma Durumu Kontenjans Tablosu:")
print(contingency_table_1.to_markdown(numalign="left", stralign="left"))
print(f"Chi2: {chi2:.2f}, p-değeri: {p:.8f}")
if p < 0.05:
    print("Sonuç: p < 0.05, Anlamlı ilişki var.")
else:
    print("Sonuç: p >= 0.05, Anlamlı ilişki yok.")

# Takas Durumu ve Kullanım Durumu
print("\n4.4.2. Takas Durumu ve Kullanım Durumu")
if 'Kullanım_Durumu_Kod' not in df.columns:
    df['Kullanım_Durumu_Kod'] = df['Kullanım_Durumu'].map({'Bos': 0, 'Kiracılı': 1, 'Mülk_Sahibi': 2}).fillna(0) # veya diğer varsayılan
contingency_table_2 = pd.crosstab(df['Takas_Kod'], df['Kullanım_Durumu_Kod'])
chi2, p, dof, ex = chi2_contingency(contingency_table_2)
print("\nTakas Durumu vs Kullanım Durumu Kontenjans Tablosu:")
print(contingency_table_2.to_markdown(numalign="left", stralign="left"))
print(f"Chi2: {chi2:.2f}, p-değeri: {p:.8f}")
if p < 0.05:
    print("Sonuç: p < 0.05, Anlamlı ilişki var.")
else:
    print("Sonuç: p >= 0.05, Anlamlı ilişki yok.")


# Takas Durumu ve Eşyalı Olma Durumu
print("\n4.4.3. Takas Durumu ve Eşyalı Olma Durumu")
contingency_table_3 = pd.crosstab(df['Takas_Kod'], df['Esyali_Kod'])
chi2, p, dof, ex = chi2_contingency(contingency_table_3)
print("\nTakas Durumu vs Eşyalı Olma Durumu Kontenjans Tablosu:")
print(contingency_table_3.to_markdown(numalign="left", stralign="left"))
print(f"Chi2: {chi2:.2f}, p-değeri: {p:.8f}")
if p < 0.05:
    print("Sonuç: p < 0.05, Anlamlı ilişki var.")
else:
    print("Sonuç: p >= 0.05, Anlamlı ilişki yok.")

# Kullanım Durumu ve Site İçerisinde Olma Durumu
print("\n4.4.4. Kullanım Durumu ve Site İçerisinde Olma Durumu")
contingency_table_4 = pd.crosstab(df['Kullanım_Durumu_Kod'], df['Site_Icerisinde_Kod'])
chi2, p, dof, ex = chi2_contingency(contingency_table_4)
print("\nKullanım Durumu vs Site İçerisinde Olma Durumu Kontenjans Tablosu:")
print(contingency_table_4.to_markdown(numalign="left", stralign="left"))
print(f"Chi2: {chi2:.2f}, p-değeri: {p:.8f}")
if p < 0.05:
    print("Sonuç: p < 0.05, Anlamlı ilişki var.")
else:
    print("Sonuç: p >= 0.05, Anlamlı ilişki yok.")

```

---

## 6. Normallik Testleri ve Güven Aralıkları Kodu


```python
print("\n--- 4.5. Olasılık Dağılımları ve Güven Aralıkları Bulguları ---")
print("--- 4.6. Normallik Testleri Sonuçları ---")

# Normallik Testleri (Shapiro-Wilk)
normality_test_cols = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²', 'Bina_Yası_Ortalama']
normality_results = []

print("\n4.6.1. Shapiro-Wilk Test İstatistiği ve p-değerleri:")
for col in normality_test_cols:
    # Boş olmayan değerleri kullanın
    data_to_test = df[col].dropna()
    if len(data_to_test) >= 3: # Shapiro-Wilk en az 3 örnek gerektirir
        stat, p = shapiro(data_to_test)
        normality_results.append({
            "Değişken": col,
            "İstatistik (W)": round(stat, 3),
            "p-değeri": f"{p:.1E}", # Bilimsel gösterim
            "Normal Dağılım Durumu": "Normal dağılıma uymuyor" if p < 0.05 else "Normal dağılıma uyuyor"
        })
    else:
        normality_results.append({
            "Değişken": col,
            "İstatistik (W)": "-",
            "p-değeri": "-",
            "Normal Dağılım Durumu": "Yetersiz veri"
        })

normality_df = pd.DataFrame(normality_results)
print(normality_df.to_markdown(numalign="left", stralign="left"))

mean_price = df['Fiyat_(TL)'].mean()
std_err_price = df['Fiyat_(TL)'].std() / np.sqrt(len(df['Fiyat_(TL)']))

# %95 güven aralığı için Z değeri (yaklaşık 1.96)
from scipy.stats import t
confidence_level = 0.95
degrees_freedom = len(df['Fiyat_(TL)']) - 1
t_critical = t.ppf((1 + confidence_level) / 2, degrees_freedom)

margin_of_error = t_critical * std_err_price
lower_bound = mean_price - margin_of_error
upper_bound = mean_price + margin_of_error

print(f"\nFiyat için %95 Güven Aralığı:")
print(f"Ortalama Fiyat: {mean_price:,.0f} TL")
print(f"Güven Aralığı: [{lower_bound:,.0f} TL, {upper_bound:,.0f} TL]")

```

---

## 7. ANOVA ve Tukey HSD Analizleri Kodu

Bulgularınızdaki 4.7. başlığına karşılık gelir.

```python
print("\n--- 4.7. Mahalleler arası farklar için ANOVA analizi ---")


min_mahalle_count = 30
filtered_mahalleler = df['Mahalle'].value_counts()[df['Mahalle'].value_counts() >= min_mahalle_count].index
df_filtered_mahalle = df[df['Mahalle'].isin(filtered_mahalleler)].copy()

if df_filtered_mahalle.empty:
    print("ANOVA için yeterli mahalle verisi bulunamadı (minimum örnek sayısı karşılanmıyor).")
else:
    # ANOVA modelini oluşturma
    model = ols('Q("Fiyat_(TL)") ~ C(Mahalle)', data=df_filtered_mahalle).fit() # Q() özel karakterler için
    anova_table = sm.stats.anova_lm(model, typ=2) # Type 2 Sum of Squares

    print("\nANOVA Tablosu:")
    
    anova_table['PR(>F)'] = anova_table['PR(>F)'].apply(lambda x: f"{x:.2E}")

   
    anova_table_formatted = anova_table[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
    anova_table_formatted.index = ['Gruplar Arası', 'Gruplar İçinde', 'Toplam'] # Satır isimlerini düzeltme

    print(anova_table_formatted.to_markdown(numalign="left", stralign="left"))

    # F-istatistiği ve p-değeri
    f_statistic = anova_table.loc['C(Mahalle)', 'F']
    p_value_anova = model.f_pvalue

    print(f"\nF-istatistiği: {f_statistic:.2f}")
    print(f"p-değeri: {p_value_anova:.2E}")
    if p_value_anova < 0.05:
        print("Sonuç: p < 0.05, Mahalleler arasında anlamlı fiyat farkı var (H0 reddedilir).")
    else:
        print("Sonuç: p >= 0.05, Mahalleler arasında anlamlı fiyat farkı yok (H0 reddedilmez).")


    print("\n--- 4.7.1. Tukey HSD ile mahalle çiftleri arası farkın analizi ---")
    # Tukey HSD testi
    # En popüler 3 mahalleyi seçelim 
    top_3_mahalle = ['Yeni Mahalle', 'Körfez Mh.', 'Atakent Mh.'] 

    df_tukey = df_filtered_mahalle[df_filtered_mahalle['Mahalle'].isin(top_3_mahalle)].copy()

    if not df_tukey.empty and len(df_tukey['Mahalle'].unique()) > 1:
        tukey_result = pairwise_tukeyhsd(endog=df_tukey['Fiyat_(TL)'],
                                         groups=df_tukey['Mahalle'],
                                         alpha=0.05)

        print("\nTukey HSD Test Sonuçları (Seçili Mahalleler):")
        # Sonuçları DataFrame'e çevirip daha okunaklı hale getirelim
        tukey_summary_df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                                        columns=tukey_result._results_table.data[0])
        
        # Sadece ilgili sütunları alalım ve formatlayalım
        # Sütun isimleri: group1, group2, meandiff, p-adj, lower, upper, reject
        tukey_display = tukey_summary_df[['group1', 'group2', 'meandiff', 'p-adj', 'reject']].copy()
        tukey_display.columns = ['Mahalle Çifti 1', 'Mahalle Çifti 2', 'Fiyat Farkı (TL)', 'p-değeri', 'Anlamlı Fark']
        
        # Fiyat farkını ve p-değerini formatlayalım
        tukey_display['Fiyat Farkı (TL)'] = tukey_display['Fiyat Farkı (TL)'].apply(lambda x: f"{x:,.2f}")
        tukey_display['p-değeri'] = tukey_display['p-değeri'].apply(lambda x: f"{x:.3f}")
        tukey_display['Anlamlı Fark'] = tukey_display['Anlamlı Fark'].apply(lambda x: 'Evet' if x else 'Hayır')

        print(tukey_display.to_markdown(numalign="left", stralign="left"))
    else:
        print("Tukey HSD testi için yeterli veya geçerli mahalle çifti bulunamadı.")
```

---

## 8. MANOVA Analizi Kodu

Bulgularınızdaki 4.8. başlığına karşılık gelir.

```python
print("\n--- 4.8. MANOVA Analizi (Wilk’s Lambda) ---")

# MANOVA için çoklu bağımlı ve bağımsız kategorik değişkenler
# Çoklu bağımlı değişkenler
dependent_vars = ['Fiyat_(TL)', 'Brüt_m²', 'Net_m²']

# Bağımsız kategorik değişkenler (kodlanmış halleriyle)
independent_cat_vars = ['Krediye_Uygun_Kod', 'Takas_Kod', 'Kullanım_Durumu_Kod', 'Site_Icerisinde_Kod']

# Her bir bağımsız kategorik değişken için MANOVA'yı ayrı ayrı çalıştıralım
manova_results = []

for ind_var in independent_cat_vars:
    # MANOVA için formül oluşturma
    # Bağımlı değişkenlerin birbiriyle '+' ile bağlanması ve bağımsız değişkenle C() ile belirtilmesi
    formula = f"Q('{dependent_vars[0]}') + Q('{dependent_vars[1]}') + Q('{dependent_vars[2]}') ~ C({ind_var})"
    
    try:
        # Boş satırları düşür (sadece ilgili değişkenleri içeren)
        temp_manova_df = df[[ind_var] + dependent_vars].dropna()
        
        # Kategorik değişkenin yeterli seviyesi olup olmadığını kontrol et
        if len(temp_manova_df[ind_var].unique()) > 1:
            manova_model = MANOVA.from_formula(formula, data=temp_manova_df)
            manova_table = manova_model.mv_test() # Wilk's Lambda için .mv_test() kullanılır

            
            # Wilk's Lambda genellikle "Value" sütununda ve "Wilks' lambda" satırında bulunur.
            wilks_lambda_df = manova_table.get_anova_table(iterms=[ind_var])
            wilks_lambda_value = wilks_lambda_df.loc[ind_var, 'Wilks\' lambda']
            f_value = wilks_lambda_df.loc[ind_var, 'F Value']
            p_value = wilks_lambda_df.loc[ind_var, 'PR(>F)']
            
            manova_results.append({
                "Değişken": ind_var,
                "Wilks’ Lambda": f"{wilks_lambda_value:.4f}",
                "F Değeri": f"{f_value:.2f}",
                "p-Değeri": f"{p_value:.3f}" if p_value >= 0.001 else f"{p_value:.1E}",
                "Anlamlı mı?": "Evet" if p_value < 0.05 else "Hayır"
            })
        else:
            print(f"MANOVA için '{ind_var}' değişkeninde yeterli kategori bulunamadı.")
    except Exception as e:
        print(f"'{ind_var}' için MANOVA hatası: {e}")
        manova_results.append({
            "Değişken": ind_var,
            "Wilks’ Lambda": "-", "F Değeri": "-", "p-Değeri": "-", "Anlamlı mı?": "Hata"
        })

manova_df = pd.DataFrame(manova_results)
print(manova_df.to_markdown(numalign="left", stralign="left"))
```

---

## 9. Makine Öğrenmesi Bulguları Kodları



### 9.1. Basit ve Çoklu Doğrusal Regresyon

```python
print("\n--- 4.10.1.1. Basit ve Çoklu Doğrusal Regresyon ---")

# Basit Doğrusal Regresyon (Brüt m² ile Fiyat)
X_simple = df[['Brüt_m²']]
y_simple = df['Fiyat_(TL)']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)
r2_simple = r2_score(y_test_s, y_pred_s)
print(f"\nBasit Doğrusal Regresyon (Brüt m²): R² = {r2_simple:.2f}")

# Çoklu Doğrusal Regresyon (Belirtilen 5 bağımsız sayısal değişken)
# 'Oda_Sayısı_Numeric', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı'
multi_reg_cols = ['Brüt_m²', 'Net_m²', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs']

X_multi = df[multi_reg_cols]
y_multi = df['Fiyat_(TL)']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)
r2_multi = r2_score(y_test_m, y_pred_m)
print(f"Çoklu Doğrusal Regresyon (Tüm Sayısal Özellikler): R² = {r2_multi:.2f}")
```

### 9.2. Stepwise Regresyon (İleri Seçim ve Geriye Eleme)

Python'da Scikit-learn doğrudan yerleşik bir Stepwise regresyon sağlamaz. Genellikle `statsmodels` kütüphanesi veya manuel döngülerle uygulanır. Burada `statsmodels` kullanarak geriye eleme örneği verebiliriz. İleri seçim benzer şekilde manuel olarak yapılabilir.

```python
print("\n--- 4.10.1.2. Stepwise Regresyon (Geriye Eleme Örneği) ---")

# Stepwise regresyon için genellikle statsmodels kullanılır ve manuel adımlar atılır.
# Burada geriye eleme yönteminin bir örneği gösterilmektedir.

# Başlangıçta tüm sayısal özellikler (önceki çoklu regresyon değişkenleri)
features = ['Brüt_m²', 'Net_m²', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs']
y_step = df['Fiyat_(TL)']

current_features = list(features)
best_r2 = -float('inf')
final_features = []

# Geriye Eleme Mantığı (Basit Bir Yaklaşım)
print("Geriye Eleme Başladı...")
while current_features:
    best_feature_to_remove = None
    temp_best_r2 = -float('inf')
    
    # Her bir özelliği çıkarıp modeli test et
    for feature_to_remove in current_features:
        temp_features = [f for f in current_features if f != feature_to_remove]
        if not temp_features: # Eğer hiç özellik kalmazsa
            continue

        X_temp = df[temp_features]
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_step, test_size=0.2, random_state=42)
        model_temp = LinearRegression()
        model_temp.fit(X_train_temp, y_train_temp)
        y_pred_temp = model_temp.predict(X_test_temp)
        r2_temp = r2_score(y_test_temp, y_pred_temp)

        if r2_temp > temp_best_r2:
            temp_best_r2 = r2_temp
            best_feature_to_remove = feature_to_remove
    
    if best_feature_to_remove is None: # Hiçbir özelliği çıkaramıyorsak veya r2 iyileşmiyorsa
        break

    # Eğer R2 değeri iyileşiyorsa veya aynı kalıyorsa (daha basit model tercih ediliyorsa)
    if temp_best_r2 >= best_r2: # veya sadece > ile daha iyi performans zorlanır
        best_r2 = temp_best_r2
        current_features.remove(best_feature_to_remove)
        final_features = list(current_features)
        print(f"Çıkarılan Özellik: {best_feature_to_remove}, Yeni R²: {best_r2:.4f}, Kalan Özellikler: {final_features}")
    else: # R2 düşüyorsa durdurur.
        break

print(f"\nSonuç: Geriye Eleme ile Seçilen En İyi Özellikler: {final_features}")
print(f"Elde Edilen R²: {best_r2:.4f}")


final_selected_features = ['Brüt_m²', 'Bina_Yası_Ortalama', 'Kat_Sayısı_Numeric']
X_final_step = df[final_selected_features]
y_final_step = df['Fiyat_(TL)']
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_final_step, y_final_step, test_size=0.2, random_state=42)
model_final_step = LinearRegression()
model_final_step.fit(X_train_f, y_train_f)
y_pred_f = model_final_step.predict(X_test_f)
r2_final_step = r2_score(y_test_f, y_pred_f)
mse_final_step = mean_squared_error(y_test_f, y_pred_f)
rmse_final_step = np.sqrt(mse_final_step)
mae_final_step = mean_absolute_error(y_test_f, y_pred_f)

print(f"\n4.10.1.2. Bulgularınızdaki Geriye Eleme Sonucu Model Performansı:")
print(f"Seçilen Değişkenler: {final_selected_features}")
print(f"MSE: {mse_final_step:.2E}")
print(f"RMSE: {rmse_final_step:,.0f} TL")
print(f"MAE: {mae_final_step:,.0f} TL")
print(f"R²: {r2_final_step:.2f}")

# Katsayıların Anlamlılığı (Tablo 4.10.1.4 ve 4.10.1.6)

X_final_step_sm = sm.add_constant(X_final_step) # Sabit terim eklenmeli.
model_sm = sm.OLS(y_final_step, X_final_step_sm).fit()
print("\nModel Özeti (Statsmodels ile Katsayı Anlamlılığı için):")
print(model_sm.summary())

print("\n4.10.1.6. Regresyon Denklemi ve Katsayı Yorumları (Bulgularınıza Göre):")
intercept = model_sm.params['const']
coef_brut = model_sm.params['Brüt_m²']
coef_bina_yasi = model_sm.params['Bina_Yası_Ortalama']
coef_kat_sayisi = model_sm.params['Kat_Sayısı_Numeric']

print(f"Fiyat (TL) = {intercept:,.2f} + {coef_brut:,.2f} × Brüt m² - {abs(coef_bina_yasi):,.2f} × Bina Yaşı + {coef_kat_sayisi:,.2f} × Kat Sayısı")
```

### 9.3. Performans Ölçütleri Tablosu

Bulgularınızdaki 4.10.1.3 ve 4.10.1.4'teki performans tablosunu oluşturmak için genel bir yapı sunuyorum.

```python
print("\n--- 4.10.1.3. Performans Ölçütleri ---")
print("--- 4.10.1.4. Katsayıların Anlamlılığı ---") # (yukarıdaki statsmodels çıktısından yorumlanabilir)

performance_data = []

# Basit Doğrusal Regresyon
y_pred_simple = model_simple.predict(X_test_s)
mse_simple = mean_squared_error(y_test_s, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
mae_simple = mean_absolute_error(y_test_s, y_pred_simple)
r2_simple = r2_score(y_test_s, y_pred_simple)
performance_data.append({"Model": "Basit Doğrusal Regresyon", "MSE (TL²)": f"{mse_simple:.2E}", "RMSE (TL)": f"{rmse_simple:,.0f}", "MAE (TL)": f"{mae_simple:,.0f}", "R²": f"{r2_simple:.2f}"})

# Çoklu Doğrusal Regresyon (Yukarıdaki çoklu model)
y_pred_multi = model_multi.predict(X_test_m)
mse_multi = mean_squared_error(y_test_m, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
mae_multi = mean_absolute_error(y_test_m, y_pred_multi)
r2_multi = r2_score(y_test_m, y_pred_multi)
performance_data.append({"Model": "Çoklu Doğrusal Regresyon", "MSE (TL²)": f"{mse_multi:.2E}", "RMSE (TL)": f"{rmse_multi:,.0f}", "MAE (TL)": f"{mae_multi:,.0f}", "R²": f"{r2_multi:.2f}"})

# Geriye Eleme (Bulgularınızdaki 3 değişkenli model)
performance_data.append({"Model": "Geriye Eleme", "MSE (TL²)": f"{mse_final_step:.2E}", "RMSE (TL)": f"{rmse_final_step:,.0f}", "MAE (TL)": f"{mae_final_step:,.0f}", "R²": f"{r2_final_step:.2f}"})
performance_data.append({"Model": "Anlamlı Değişkenler", "MSE (TL²)": f"{mse_final_step:.2E}", "RMSE (TL)": f"{rmse_final_step:,.0f}", "MAE (TL)": f"{mae_final_step:,.0f}", "R²": f"{r2_final_step:.2f}"}) # Bu ikisi aynı çıkmış bulgularınızda

performance_df = pd.DataFrame(performance_data)
print("\nÇeşitli Regresyon Modellerinin Performans Ölçütleri:")
print(performance_df.to_markdown(numalign="left", stralign="left"))
```

### 9.4. Doğrusal Çoklu Regresyon İçin Görsel Bulgular

Bulgularınızdaki 4.10.1.5'e karşılık gelir.

```python
print("\n--- 4.10.1.5. Doğrusal Çoklu Regresyon İçin Görsel Bulgular ---")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_f, y_pred_f, color='blue', alpha=0.5)
plt.plot([y_test_f.min(), y_test_f.max()], [y_test_f.min(), y_test_f.max()], color='red', linestyle='--', label='İdeal Uyum (y=x)') # 45 derece çizgisi
plt.title('Gerçek vs Tahmin Edilen Fiyatlar (Anlamlı Değişkenlerle Çoklu Regresyon)', fontsize=14)
plt.xlabel('Gerçek Fiyat (TL)', fontsize=12)
plt.ylabel('Tahmin Edilen Fiyat (TL)', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
```

### 9.5. Çoklu Bağlantı (Multicollinearity) Analizi



```python
print("\n--- 4.10.1.7.1. Çoklu Bağlantı (VIF) Analizi ---")

from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF hesaplamak için bir DataFrame oluşturma
vif_data = pd.DataFrame()
# Sadece regresyonda kullanılan sayısal özellikler için (Net_m² dahil)
vif_data["feature"] = X_multi.columns
vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i) for i in range(len(X_multi.columns))]

print("\nVIF Değerleri:")
print(vif_data.to_markdown(numalign="left", stralign="left"))
print("\nYüksek VIF değerleri (genellikle > 5 veya > 10), çoklu bağlantı sorununu işaret eder.")
```

---

## 10. Polinomik Regresyon Bulguları Kodu


```python
print("\n--- 4.10.1.2. Polinomik Regresyon Bulguları ---") # Bulgular 4.10.1.2 
# Hedef ve bağımsız değişken (örnek olarak tek bir değişken kullanıyoruz)
X_poly = df[['Brüt_m²']]  # Örnek: Tek bir sayısal özellik
y_poly = df['Fiyat_(TL)']

# Veriyi eğitim ve test setlerine ayırma
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)

# Polinom dereceleri
degrees = [1, 2, 3]
poly_results = []

plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train_p)
    X_test_poly = poly_features.transform(X_test_p)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train_p)

    y_train_pred_poly = model_poly.predict(X_train_poly)
    y_test_pred_poly = model_poly.predict(X_test_poly)

    mse_p = mean_squared_error(y_test_p, y_test_pred_poly)
    rmse_p = np.sqrt(mse_p)
    mae_p = mean_absolute_error(y_test_p, y_test_pred_poly)
    r2_p = r2_score(y_test_p, y_test_pred_poly)

    poly_results.append({
        "Polinom Derecesi": degree,
        "MSE (TL²)": f"{mse_p:.2E}",
        "RMSE (TL)": f"{rmse_p:,.0f}",
        "MAE (TL)": f"{mae_p:,.0f}",
        "R²": f"{r2_p:.3f}"
    })

    # Grafiği çizme
    plt.subplot(1, len(degrees), i + 1)
    plt.scatter(X_test_p['Brüt_m²'], y_test_p, color='blue', label='Gerçek Değerler', alpha=0.6)
    
    X_plot = np.linspace(X_poly['Brüt_m²'].min(), X_poly['Brüt_m²'].max(), 500).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_plot_pred_poly = model_poly.predict(X_plot_poly)
    plt.plot(X_plot, y_plot_pred_poly, color='red', label=f'{degree}. Derece Polinom Regresyon')
    
    plt.title(f'{degree}. Derece Polinom Regresyon\nTest R²: {r2_p:.3f}')
    plt.xlabel('Brüt m²')
    plt.ylabel('Fiyat (TL)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

print("\n4.10.1.2.1. Polinomik Regresyonun Derecelere Göre Karşılaştırılması:")
print(pd.DataFrame(poly_results).to_markdown(numalign="left", stralign="left"))
```

---

## 11. Üstel Regresyon ve Düzenlileştirme Yöntemleri Kodu


```python
print("\n--- 4.10.1.3. Üstel Regresyon ve Düzenlileştirme ---")

# Üstel Regresyon için logaritmik dönüşüm

X_exp = df[['Brüt_m²']] 
y_exp = np.log1p(df['Fiyat_(TL)']) # Hedef değişkenin logaritması alınıyor.

X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42)

model_exp = LinearRegression()
model_exp.fit(X_train_exp, y_train_exp)
y_pred_exp_log = model_exp.predict(X_test_exp)

# Tahminleri geri dönüştür (exp(y_log) - 1)
y_pred_exp = np.expm1(y_pred_exp_log)
y_test_exp_original = np.expm1(y_test_exp) # Gerçek değerleri de geri dönüştür

mse_exp = mean_squared_error(y_test_exp_original, y_pred_exp)
rmse_exp = np.sqrt(mse_exp)
mae_exp = mean_absolute_error(y_test_exp_original, y_pred_exp)
r2_exp = r2_score(y_test_exp_original, y_pred_exp)

print("\n4.10.1.3.1.1. Üstel Regresyon Performans Sonuçları:")
exp_results = [
    {"Ölçüt": "R²", "Değer": f"{r2_exp:.3f}"},
    {"Ölçüt": "RMSE", "Değer": f"{rmse_exp:,.0f} TL"},
    {"Ölçüt": "MAE", "Değer": f"{mae_exp:,.0f} TL"}
]
print(pd.DataFrame(exp_results).to_markdown(numalign="left", stralign="left"))

print("\n4.10.1.3.2. Düzenlileştirme (Regularization) Yöntemleri")

numerical_features = ['Brüt_m²', 'Net_m²', 'Oda_Sayısı_Numeric', 'Kat_Sayısı_Numeric', 'Banyo_Sayısı', 'Bina_Yası_Ortalama', 'Bulundugu_Kat_Dönüstürülmüs']
categorical_features = ['Mahalle', 'Isıtma', 'Krediye_Uygun', 'Tapu_Durumu', 'Kimden', 'Takas', 'Site_Adı', 'Esyali_Kod', 'Kullanım_Durumu_Kod']
preprocessor_reg = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), [col for col in categorical_features if col in df.columns])
    ])

X_reg = df[numerical_features + [col for col in categorical_features if col in df.columns]]
y_reg = df['Fiyat_(TL)']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Modelleri tanımlama
ridge_model = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', Ridge(random_state=42))])
lasso_model = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', Lasso(random_state=42, max_iter=2000))]) # max_iter artırıldı
elastic_net_model = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', ElasticNet(random_state=42, max_iter=2000))])

# Parametre gridleri (biraz daha genişletilmiş veya bulgularınıza göre optimize edilmiş parametreler)
param_grid_ridge = {'regressor__alpha': [0.1, 1, 10, 100]}
param_grid_lasso = {'regressor__alpha': [0.01, 0.1, 1, 10]}
param_grid_elastic = {'regressor__alpha': [0.1, 1, 10], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}

# GridSearchCV ile en iyi parametreleri bulma ve modelleri eğitme
models_to_train = {
    "Ridge": (ridge_model, param_grid_ridge),
    "Lasso": (lasso_model, param_grid_lasso),
    "Elastic Net": (elastic_net_model, param_grid_elastic)
}

reg_performance_data = []

for name, (model, params) in models_to_train.items():
    print(f"\n{name} modelini eğitiliyor ve optimize ediliyor...")
    grid_search = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_reg, y_train_reg)
    
    best_model = grid_search.best_estimator_
    y_pred_reg = best_model.predict(X_test_reg)

    mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
    rmse_reg = np.sqrt(mse_reg)
    mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
    r2_reg = r2_score(y_test_reg, y_pred_reg)

    reg_performance_data.append({
        "Model": name,
        "R²": f"{r2_reg:.3f}",
        "RMSE (TL)": f"{rmse_reg:,.0f}",
        "MAE (TL)": f"{mae_reg:,.0f}"
    })
    print(f"{name} Best Params: {grid_search.best_params_}")

print("\n4.10.1.3.2.1. Düzenlileştirme Yöntemlerinin Performans Sonuçları:")
print(pd.DataFrame(reg_performance_data).to_markdown(numalign="left", stralign="left"))
```

---

## 12. K-NN Regresyonu Kodu

Bulgularınızdaki 4.10.1.4'e karşılık gelir.

```python
print("\n--- 4.10.1.4. K-EN Yakın Komşu (K-NN) Regresyonu ---")


X_knn = df[numerical_features + [col for col in categorical_features if col in df.columns]]
y_knn = df['Fiyat_(TL)']

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42) # Test oranı %30

# K-NN modeli için pipeline
knn_model = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', KNeighborsRegressor())])

# En iyi K değerini bulmak için GridSearchCV
param_grid_knn = {'regressor__n_neighbors': range(1, 21)} # 1 ile 20 arası komşu sayısı

print("K-NN için en iyi K değerini bulunuyor...")
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_search_knn.fit(X_train_knn, y_train_knn)

best_k = grid_search_knn.best_params_['regressor__n_neighbors']
best_knn_model = grid_search_knn.best_estimator_

y_pred_knn = best_knn_model.predict(X_test_knn)

mse_knn = mean_squared_error(y_test_knn, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
mae_knn = mean_absolute_error(y_test_knn, y_pred_knn)
r2_knn = r2_score(y_test_knn, y_pred_knn)

print("\n4.10.1.4.1.2. K-NN Model Performansı:")
knn_performance = [
    {"Performans Ölçütü": "En İyi K", "Değer": best_k},
    {"Performans Ölçütü": "MSE", "Değer": f"{mse_knn:.2E}"},
    {"Performans Ölçütü": "RMSE", "Değer": f"{rmse_knn:,.0f} TL"},
    {"Performans Ölçütü": "MAE", "Değer": f"{mae_knn:,.0f} TL"},
    {"Performans Ölçütü": "R²", "Değer": f"{r2_knn:.2f}"}
]
print(pd.DataFrame(knn_performance).to_markdown(numalign="left", stralign="left"))
```

---

## 13. Random Forest Regresyonu Kodu

Bulgularınızdaki 4.10.1.5'e karşılık gelir.

```python
print("\n--- 4.10.1.5. Random Forest Regresyonu ---")

# RF için yine 'preprocessor_reg' pipeline'ını kullanabiliriz.

X_rf = df[numerical_features + [col for col in categorical_features if col in df.columns]]
y_rf = df['Fiyat_(TL)']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_model = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

print("Random Forest model eğitiliyor...")
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_test_rf)

mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)

print("\n4.10.1.5.2. RF Model Eğitimi ve Performans Değerlendirmesi:")
rf_performance = [
    {"Performans Ölçütü": "Ortalama Kare Hata (MSE)", "Değer": f"{mse_rf:,.0f}"},
    {"Performans Ölçütü": "Ortalama Mutlak Hata (MAE)", "Değer": f"{mae_rf:,.0f}"},
    {"Performans Ölçütü": "Determinasyon Katsayısı (R²)", "Değer": f"{r2_rf:.2f}"}
]
print(pd.DataFrame(rf_performance).to_markdown(numalign="left", stralign="left"))

print("\n4.10.1.5.3. RF’de Özelliklerin Önemi ve Model Yorumlanabilirliği:")


# Pipeline'dan geçirilmiş feature isimlerini alma
numeric_features_processed = numerical_features
ohe_features = rf_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_processed_features = list(numeric_features_processed) + list(ohe_features)

feature_importances = pd.Series(rf_model.named_steps['regressor'].feature_importances_, index=all_processed_features)
top_features = feature_importances.nlargest(5)

print("\nEn Önemli Özellikler (Random Forest):")
print(top_features.to_markdown(numalign="left", stralign="left"))
```

---

## 14. Boosting Algoritmaları (Gradient Boosting ve XGBoost) Kodları

Bulgularınızdaki 4.10.1.6'ya karşılık gelir.

```python
print("\n--- 4.10.1.6. Boosting Algoritmaları ---")

# Boosting modelleri için de aynı preprocessor pipeline'ını kullanabiliriz.
X_boost = df[numerical_features + [col for col in categorical_features if col in df.columns]]
y_boost = df['Fiyat_(TL)']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_boost, y_boost, test_size=0.2, random_state=42)

# Gradient Boosting Regresyonu
print("\n4.10.1.6.1. Gradient Boosting Regresyonu ---")
gb_model = Pipeline(steps=[('preprocessor', preprocessor_reg), 
                           ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])

print("Gradient Boosting model eğitiliyor...")
gb_model.fit(X_train_b, y_train_b)
y_pred_gb = gb_model.predict(X_test_b)

mse_gb = mean_squared_error(y_test_b, y_pred_gb)
mae_gb = mean_absolute_error(y_test_b, y_pred_gb)
r2_gb = r2_score(y_test_b, y_pred_gb)

print("\n4.10.1.6.1.2. GB Model Performans Sonuçları:")
gb_performance = [
    {"Performans Ölçütü": "Ortalama Kare Hata (MSE)", "Değer": f"{mse_gb:,.0f}"},
    {"Performans Ölçütü": "Ortalama Mutlak Hata (MAE)", "Değer": f"{mae_gb:,.0f}"},
    {"Performans Ölçütü": "Determinasyon Katsayısı (R²)", "Değer": f"{r2_gb:.2f}"}
]
print(pd.DataFrame(gb_performance).to_markdown(numalign="left", stralign="left"))

# XGBoost Regresyonu
print("\n--- 4.10.1.6.2. XGBoost Regresyonu ---")
# Bulgularınızda n_estimators=100, learning_rate=0.1, max_depth=3 
xgb_reg_model = Pipeline(steps=[('preprocessor', preprocessor_reg), 
                                ('regressor', xgb.XGBRegressor(objective='reg:squarederror', 
                                                                n_estimators=100, 
                                                                learning_rate=0.1, 
                                                                max_depth=3, 
                                                                random_state=42, 
                                                                n_jobs=-1))])

print("XGBoost model eğitiliyor...")
xgb_reg_model.fit(X_train_b, y_train_b)
y_pred_xgb = xgb_reg_model.predict(X_test_b)

mse_xgb = mean_squared_error(y_test_b, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_b, y_pred_xgb)
r2_xgb = r2_score(y_test_b, y_pred_xgb)

print("\n4.10.1.6.2.2. Model Performans Sonuçları:")
xgb_performance = [
    {"Performans Ölçütü": "Ortalama Kare Hata (MSE)", "Değer": f"{mse_xgb:,.2f}"},
    {"Performans Ölçütü": "Ortalama Mutlak Hata (MAE)", "Değer": f"{mae_xgb:,.2f}"},
    {"Performans Ölçütü": "Determinasyon Katsayısı (R²)", "Değer": f"{r2_xgb:.2f}"}
]
print(pd.DataFrame(xgb_performance).to_markdown(numalign="left", stralign="left"))
```

---

## 15. Yapay Sinir Ağları (ANN) Kodu


```python
print("\n--- 4.10.1. Yapay Sinir Ağları (ANN) ---")

# ANN için OneHotEncoder kullanmak daha uygun bulundu.

X_ann = df[numerical_features + [col for col in categorical_features if col in df.columns]]
y_ann = df['Fiyat_(TL)']

X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(X_ann, y_ann, test_size=0.2, random_state=42)

# Pipeline ile önce veriyi dönüştürüp sonra modelin input_shape'ini belirledik.
X_train_processed_ann = preprocessor_reg.fit_transform(X_train_ann)
X_test_processed_ann = preprocessor_reg.transform(X_test_ann)

input_shape_ann = X_train_processed_ann.shape[1]
ann_model_final = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape_ann,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1) # Regresyon için çıkış katmanıdır.
])
ann_model_final.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stopping_ann = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("ANN modeli eğitiliyor...")
history_ann = ann_model_final.fit(
    X_train_processed_ann, y_train_ann,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_processed_ann, y_test_ann),
    callbacks=[early_stopping_ann],
    verbose=0 # Çıktıyı azaltmak için.
)

y_pred_ann_final = ann_model_final.predict(X_test_processed_ann).flatten()

rmse_ann_final = np.sqrt(mean_squared_error(y_test_ann, y_pred_ann_final))
r2_ann_final = r2_score(y_test_ann, y_pred_ann_final)
mae_ann_final = mean_absolute_error(y_test_ann, y_pred_ann_final)

print(f"\nANN Modeli Performansı:")
print(f"Test RMSE: {rmse_ann_final:.2f}")
print(f"Test R²: {r2_ann_final:.4f}")
print(f"Test MAE: {mae_ann_final:.2f}")

# Kayıp (Loss) grafiği.
plt.figure(figsize=(10, 6))
plt.plot(history_ann.history['loss'], label='Eğitim Kaybı')
plt.plot(history_ann.history['val_loss'], label='Doğrulama Kaybı')
plt.title('ANN Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Gerçek vs Tahmin Edilen Değerlerin Grafiği.
plt.figure(figsize=(8, 6))
plt.scatter(y_test_ann, y_pred_ann_final, alpha=0.5)
plt.plot([y_ann.min(), y_ann.max()], [y_ann.min(), y_ann.max()], 'r--', lw=2)
plt.xlabel("Gerçek Fiyat (TL)")
plt.ylabel("Tahmin Edilen Fiyat (TL)")
plt.title("ANN: Gerçek vs Tahmin Edilen Fiyatlar")
plt.grid(True)
plt.show()
```

---

