# Atakum Konut Analitiği Dashboardu

Ana proje kurulumu tamamlandıktan sonra repo kökünde çalıştırın:

```bash
streamlit run dashboard/app.py
```

Panel dört çalışma alanı içerir:

1. Mahalle, oda tipi, fiyat ve metrekare filtreli piyasa görünümü
2. Model karşılaştırması, özellik önemi ve sızıntı duyarlılığı
3. Senaryo bazlı tek konut ilan fiyatı simülatörü
4. Örneklem akışı ve veri kalitesi kontrolleri

Model performans tablolarını yenilemek için önce:

```bash
python analiz_icin_kodlar.py
```

> Tahmin aracı resmi ekspertiz veya yatırım tavsiyesi değildir; veri setindeki ilan
> fiyatı ilişkilerini özetler.
