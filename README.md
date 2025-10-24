# Konut Fiyat Tahmini – Samsun / Atakum Örneği

## Proje Hakkında

Bu proje, Samsun’un Atakum ilçesindeki konut ilanlarından elde edilen verilerle konut fiyatlarını etkileyen temel değişkenleri belirlemeyi ve farklı makine öğrenmesi algoritmalarıyla fiyat tahmini modelleri geliştirmeyi amaçlamaktadır. Lisans bitirme tezi kapsamında hazırlanmıştır.

## Özet

Konut fiyatları konum, yapı özellikleri ve çevresel faktörlerden etkilenir. Bu proje kapsamında:

* Atakum ilçesindeki konut ilanları analiz edilmiştir.
* Fiyatları etkileyen temel faktörler belirlenmiştir.
* Tanımlayıcı istatistikler, korelasyon analizleri ve aykırı değer tespiti yapılmıştır.
* Veri ön işleme adımları (log dönüşümü, normalizasyon, eksik veri yönetimi) uygulanmıştır.

## Kullanılan Yöntemler ve Araçlar

* **Programlama Dili:** Python
* **Kütüphaneler:** pandas, numpy, scikit-learn, xgboost, lightgbm, keras/tensorflow, matplotlib, seaborn
* **Makine Öğrenmesi Modelleri:** Doğrusal Regresyon, Karar Ağaçları, Random Forest, Gradient Boosting, XGBoost, LightGBM, Yapay Sinir Ağı (ANN)

## Model Sonuçları

* En yüksek doğruluk sağlayan modeller: Random Forest, XGBoost ve ANN
* Model performansı hiperparametre optimizasyonu ve iyileştirmelerle artırılmıştır
* Bulgular, yerel emlak yatırımları ve bölgesel veri odaklı analizler açısından yol göstericidir

## Dosya Yapısı

```
Bitirme-Tezi/
│
├── README.md                  # Proje açıklaması
├── bitirmetezi.pdf            # Bitirme tezi raporu
├── veriseti.xlsx              # Emlak verisi
├── analiz_icin_kodlar.py      # Veri analizi scripti
└── Zaman Serisi.docx          # Zaman serisi analizi raporu
```

## Kurulum ve Kullanım

1. Repo klonlanır:

```bash
git clone https://github.com/Blacksidemre/Bitirme-Tezi.git
cd Bitirme-Tezi
```

2. Gerekli kütüphaneler yüklenir:

```bash
pip install -r requirements.txt
```

3. Veri analizi için:

```bash
python analiz_icin_kodlar.py
```

4. Makine öğrenmesi modellerini çalıştırmak için ilgili scriptleri kullanabilirsiniz.

## Anahtar Kelimeler

Konut Fiyat Tahmini, Makine Öğrenmesi, İstatistik, Random Forest, XGBoost, Yapay Sinir Ağı

