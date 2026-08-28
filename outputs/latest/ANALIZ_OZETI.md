# Analiz Özeti

Bu rapor, `veriseti.xlsx` dosyasından otomatik olarak üretilmiştir.

## Veri kapsamı

- Çalışmada bildirilen ham örneklem: **2.836** kayıt
- Repodaki temizlenmiş veri: **2.281** kayıt
- Yapısal olarak geçerli görüntü: **2.272** kayıt
- Tekil ilanların son görüntüsü: **1.922** ilan
- İncelenen tarih aralığı: **2024-06-23 - 2025-05-19**

## Tahmin modeli

- Seçilen model: **Extra Trees**
- Kilitli test MAE: **484.507 TL**
- Kilitli test RMSE: **695.408 TL**
- Kilitli test R²: **0.793**
- MAE için bootstrap %95 güven aralığı:
  **436.336 TL - 535.998 TL**

## Yorum sınırı

Sonuçlar gerçekleşen satış bedellerini değil, veri setindeki ilan fiyatlarını açıklar.
Model çıktıları otomatik ekspertiz veya yatırım tavsiyesi değildir. Aynı ilanların farklı
tarihlerdeki görüntüleri ana model değerlendirmesinde tekilleştirilmiştir.
