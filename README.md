
# Konut Fiyat Tahmini – Samsun / Atakum Örneği

Bu çalışma, Samsun'un Atakum ilçesindeki konut ilanlarından elde edilen verilerle konut fiyatlarını etkileyen temel değişkenleri belirlemeyi ve bu doğrultuda makine öğrenmesi algoritmalarıyla fiyat tahmini modelleri geliştirmeyi amaçlamaktadır. Proje, bir lisans bitirme tezi kapsamında hazırlanmıştır.

## Özet

Konutlar, yalnızca bireylerin barınma ihtiyacını karşılayan yapılar olmanın ötesinde, sosyal ve ekonomik hayatın önemli bir parçası olarak değerlendirilmektedir. Özellikle şehirleşmenin yoğunlaştığı bölgelerde konutların değeri; konum, yapı özellikleri ve çevresel faktörler gibi çok sayıda etkene bağlı olarak değişkenlik göstermektedir.

Bu bağlamda, Samsun'un gelişmekte olan ilçelerinden biri olan Atakum'daki konut ilanları incelenmiş; fiyatlara etki eden faktörler belirlenmiş ve bu faktörler doğrultusunda çeşitli tahmin modelleri oluşturulmuştur. Çalışma kapsamında, veri seti üzerinde tanımlayıcı istatistikler, korelasyon analizleri ve aykırı değer tespiti gibi yöntemler uygulanmış; ardından logaritmik dönüşüm gibi veri ön işleme adımları gerçekleştirilmiştir.

Makine öğrenmesi algoritmaları olarak doğrusal regresyon, rastgele orman (Random Forest) ve yapay sinir ağı (Artificial Neural Network) modelleri kullanılmış; özellikle Random Forest ve ANN modellerinin daha yüksek doğruluk sağladığı gözlemlenmiştir.

Elde edilen bulgular, hem yerel düzeyde emlak yatırımları için yol gösterici olabilir hem de bölgesel veri temelli analizlerin akademik bağlamda önemini ortaya koymaktadır.

## Yöntemler ve Kullanılan Araçlar

- Tanımlayıcı istatistiksel analizler
- Korelasyon analizi ve aykırı değer tespiti
- Veri dönüşüm teknikleri (log dönüşüm)
- Regresyon tabanlı makine öğrenmesi algoritmaları:
  - Doğrusal Regresyon
  - Random Forest
  - Yapay Sinir Ağı
- Programlama dili: Python
- Kullanılan kütüphaneler: pandas, numpy, scikit-learn, matplotlib, seaborn

## Proje Dosya Yapısı

```

bitirme-tezi/
├── rapor.pdf            # Bitirme projesi raporu
├── data/                # Veri dosyaları
├── notebooks/           # Analiz ve modelleme kodları (.ipynb)
├── models/              # Eğitilmiş modeller ve sonuç çıktıları
├── visuals/             # Grafik ve görselleştirmeler
└── README.md            # Açıklama dosyası

```

## Anahtar Kelimeler

Konut Fiyat Tahmini, Makine Öğrenmesi, İstatistiksel Analiz, Regresyon, Random Forest, Yapay Sinir Ağı, Samsun, Atakum

## Not

Bu çalışma yalnızca akademik amaçla hazırlanmış olup, ticari kullanıma açık değildir.

