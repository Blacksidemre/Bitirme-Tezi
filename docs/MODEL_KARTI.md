# Model Kartı

## Modelin amacı

Bu model ailesi, Samsun'un Atakum ilçesindeki konut **ilan fiyatları** ile ilan ve yapı
özellikleri arasındaki ilişkileri araştırmak için geliştirilmiştir. Amaç, akademik
karşılaştırma ve senaryo analizi sunmaktır; gerçekleşmiş satış bedeli, resmi ekspertiz,
kredi kararı veya yatırım getirisi tahmini üretmez.

## Sürümler ve kullanım yerleri

| Bileşen | Model | Eğitim verisi | Amaç |
|---|---|---|---|
| Akademik değerlendirme | Extra Trees | 1.537 eğitim ilanı | Modelleri adil protokolle karşılaştırmak |
| Kilitli test | Extra Trees | Eğitimden sonra 385 görülmemiş ilan | Genelleme hatasını ölçmek |
| Dashboard simülatörü | Histogram Gradient Boosting | 1.922 tekil ilanın tamamı | Hızlı, etkileşimli senaryo üretmek |

Dashboard modeli tüm veri üzerinde yeniden eğitildiğinden, arayüzdeki tekil tahminler
ayrı bir test başarısı iddiası taşımaz. Bilimsel performans raporu kilitli testteki
Extra Trees sonuçlarına dayanır.

## Girdi ve hedef

Hedef, `Fiyat (TL)` ilan talep fiyatıdır. Sayısal girdiler brüt/net alan, banyo sayısı,
bina yaşı, dönüştürülmüş kat, oda sayısı, toplam kat, aidat ve ilan günüdür. Kategorik
girdiler mahalle, ısıtma, mutfak, balkon, asansör, otopark, eşyalı olma, kullanım
durumu, site durumu, kredi uygunluğu, tapu durumu, ilan veren ve takas bilgisidir.

Sayısal eksikler medyanla doldurulur ve standartlaştırılır. Kategorik eksikler en sık
sınıfla doldurulur; nadir kategoriler birleştirilerek one-hot kodlanır. Hedef değişken
`log1p` dönüşümüyle modellenir ve tahminler TL ölçeğine geri çevrilir. Tüm dönüşümler
`scikit-learn` Pipeline yapısında, yalnızca eğitim parçasında öğrenilir.

## Model seçimi

Karşılaştırılan yöntemler:

- medyan temel model,
- Ridge regresyon,
- Random Forest,
- Extra Trees,
- Histogram Gradient Boosting.

Veri önce %80 eğitim ve %20 kilitli test olarak ayrılır. Model seçimi yalnızca eğitim
parçasındaki beş katlı çapraz doğrulama MAE değerine göre yapılır. Kilitli test, seçim
tamamlandıktan sonra bir kez raporlanır. Tüm rastgele işlemler için `random_state=42`
kullanılır.

## Performans

| Ölçüt | Sonuç |
|---|---:|
| Eğitim içi CV MAE | 468.459 ± 39.145 TL |
| Eğitim içi CV R² | 0,787 ± 0,029 |
| Kilitli test MAE | 484.507 TL |
| Kilitli test RMSE | 695.408 TL |
| Kilitli test R² | 0,793 |
| Kilitli test MAPE | %14,34 |
| Bootstrap %95 MAE aralığı | 436.336–535.998 TL |

Bootstrap aralığı test kümesindeki **ortalama mutlak hata düzeyinin** belirsizliğini
gösterir; tek bir konut için tahmin aralığı değildir.

## Sağlamlık ve sızıntı kontrolü

| Değerlendirme protokolü | MAE | R² | Yorum |
|---|---:|---:|---|
| Tekil ilanlar, kilitli rastgele test | 484.507 TL | 0,793 | Ana raporlama protokolü |
| Kronolojik kontrol | 502.892 TL | 0,764 | Geleceğe yakın kullanım için daha temkinli kontrol |
| Tekrarlı görüntüler, satır bazlı bölme | 396.731 TL | 0,829 | Aynı ilan iki tarafta kalabildiği için iyimser |
| Tekrarlı görüntüler, ilan-ID gruplu bölme | 524.420 TL | 0,693 | İlanlar kümeler arasında tamamen ayrılır |

Satır bazlı sonuç, tekrar eden ilan görüntülerinin eğitim ve testte birlikte bulunmasının
başarıyı yapay olarak yükseltebildiğini gösterir. Bu nedenle ana model bir ilan için
yalnızca en güncel görüntüyü kullanır.

## Uygun kullanım

- Atakum ilan piyasasını keşifsel olarak incelemek
- Model değerlendirme protokollerini karşılaştırmak
- Değişkenlerin tahmin hatasına katkısını araştırmak
- Veri kapsamı içinde karşılaştırmalı “ne olursa?” senaryoları üretmek
- Eğitim ve portföy gösterimi

## Uygun olmayan kullanım

- Konut satın alma veya satma kararını tek başına vermek
- Resmi ekspertiz, kredi teminat değeri veya hukuki değerleme
- Bireysel yatırım getirisi ya da garanti edilen satış fiyatı üretmek
- Atakum dışındaki konutlara doğrudan genellemek
- Girdi aralıklarının çok dışındaki konutları tahmin etmek
- Mahalle ya da kişi grupları hakkında normatif karar vermek

## Sınırlılıklar ve riskler

1. Veri, olasılıklı bir saha örneklemi değil ilan platformu örneklemidir.
2. Hedef gerçekleşmiş satış değil satıcının talep fiyatıdır.
3. Tarihler 2024-06-23 ile 2025-05-19 arasındadır ve son iki ayda yoğunlaşır.
4. Bildirilen 2.836 ham satırın dosyası repoda yoktur; önceden elenen 555 kayıt
   satır bazında yeniden denetlenemez.
5. Kesin koordinat, ulaşım süresi, manzara, cephe, deprem/zemin bilgisi ve iç kalite
   gibi önemli değişkenler bulunmaz.
6. Fiyatlar enflasyon veya başka bir fiyat endeksiyle reel hâle getirilmemiştir.
7. Özellik önemi değerleri nedensel etki ya da bağımsız fiyat primi değildir.
8. Yeni piyasa dönemlerinde veri kayması performansı düşürebilir.

## İzleme önerisi

Canlı kullanım düşünülürse veri dönemi, mahalle/oda dağılımı, medyan m² fiyatı, eksik
değer oranları ve gerçekleşmiş etiket varsa MAE aylık olarak izlenmelidir. Yeni dönem
verisi geldiğinde kronolojik bir test penceresi ayrılmalı; üretim modeli ancak önceki
sürümden daha iyi ve tutarlı sonuç verirse güncellenmelidir.

## Tekrarlanabilirlik

```bash
pip install -e ".[dashboard,dev]"
python analiz_icin_kodlar.py
pytest
```

Makine tarafından okunabilir özetler `outputs/latest/*.json`, tablo sonuçları
`outputs/latest/tables/` ve görseller `outputs/latest/figures/` altında tutulur.
