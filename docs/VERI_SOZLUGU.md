# Veri Sözlüğü

Bu belge `veriseti.xlsx` içindeki kaynak sütunları ve analiz hattında türetilen alanları
açıklar. Kaynak dosyada 2.281 satır ve 27 sütun vardır.

## Kaynak sütunlar

| Sütun | Tür | Açıklama | Örnek / not |
|---|---|---|---|
| `Fiyat (TL)` | sayısal | İlanda talep edilen toplam fiyat; hedef değişken | 2.900.000 |
| `Mahalle` | kategorik | İlanın Atakum içindeki mahallesi | Atakent Mh. |
| `İlan No` | kimlik | Platform ilan numarası | Aynı konutun görüntülerini bağlamak için kullanılır |
| `İlan Tarihi` | tarih/metin | Türkçe uzun biçimli ilan tarihi | 06 Mayıs 2025 |
| `Brüt m²` | sayısal | İlanda bildirilen brüt alan | Pozitif olmalıdır |
| `Net m²` | sayısal | İlanda bildirilen net kullanım alanı | Pozitif olmalıdır |
| `Oda Sayısı` | kategorik | İlan metnindeki oda düzeni | 2+1, 3+1 |
| `Kat Sayısı` | sayısal | Binanın toplam kat sayısı | Kaynak sayısal alan |
| `Isıtma` | kategorik | Isıtma sistemi | Kombi (Doğalgaz), Yerden Isıtma |
| `Banyo Sayısı` | sayısal | İlanda bildirilen banyo sayısı | 1–3 gözlenmiştir |
| `Mutfak` | kategorik | Mutfak tipi | Açık (Amerikan), Kapalı |
| `Balkon` | kategorik | Balkon bulunma durumu | Var, Yok |
| `Asansör` | kategorik | Binada asansör durumu | Var, Yok |
| `Otopark` | kategorik | Otopark türü veya yokluğu | Açık, Kapalı, Yok |
| `Eşyalı` | kategorik | Konutun eşyalı sunulma durumu | Evet, Hayır, Belirtilmemiş |
| `Kullanım Durumu` | kategorik | İlan anındaki kullanım durumu | Boş, Kiracılı, Mülk Sahibi |
| `Site İçerisinde` | kategorik | Konutun bir site içinde bulunması | Evet, Hayır |
| `Site Adı` | kategorik/metin | Bildirilen site adı | Ana modelde yüksek kardinalite nedeniyle kullanılmaz |
| `Aidat (TL)` | karma | İlan aidatı veya “Belirtilmemiş” değeri | Analizde sayısal türevi kullanılır |
| `Krediye Uygun` | kategorik | İlanın kredi uygunluğu beyanı | Evet, Hayır, Belirtilmemiş |
| `Tapu Durumu` | kategorik | İlanda bildirilen tapu türü | Kat Mülkiyetli vb. |
| `Kimden` | kategorik | İlanı veren taraf | Emlak Ofisinden, Sahibinden vb. |
| `Takas` | kategorik | Takas seçeneği beyanı | Evet, Hayır |
| `Bina Yaşı Ortalama` | sayısal | Yaş aralığından dönüştürülmüş yaklaşık bina yaşı | Sürekli değişken olarak kullanılır |
| `Bulunduğu Kat (Dönüştürülmüş)` | sayısal | Kat bilgisinin sayısal kodu | Bodrum katlar negatif olabilir |
| `Oda Sayısı Numeric` | sayısal | Oda düzeninden türetilmiş toplam oda bileşeni | 3+1 için 4 |
| `Kat Sayısı Numeric` | sayısal | Toplam kat bilgisinin modellemeye hazır karşılığı | Kaynakta `Kat Sayısı` ile örtüşür |

## Analiz hattında türetilen sütunlar

| Sütun | Üretim | Kullanım |
|---|---|---|
| `_Kaynak Satır` | Excel başlığı dikkate alınarak satır sırası | Hata kayıtlarını kaynağa bağlama |
| `İlan Kimliği` | `İlan No` metne çevrilip olası `.0` son eki temizlenir | Tekilleştirme ve grup bölmesi |
| `İlan Tarihi Parsed` | Türkçe ay adları ayrıştırılır | Sıralama ve kronolojik değerlendirme |
| `Aidat (TL) Numeric` | `Aidat (TL)` içindeki sayısal bölüm çıkarılır | Model girdisi; eksikler medyanla doldurulur |
| `İlan Günü` | En erken geçerli tarihten itibaren gün sayısı | Kısıtlı zaman eğilimi göstergesi |
| `Brüt m² Başına Fiyat` | `Fiyat (TL) / Brüt m²` | Tanımlayıcı analiz ve dashboard |
| `Reddetme Nedeni` | Yapısal doğrulama kurallarının açıklaması | Denetim çıktısı |

## Yapısal doğrulama kuralları

Bir satır ana analiz havuzuna alınmak için aşağıdaki koşulları sağlamalıdır:

- 8–12 basamaklı geçerli bir ilan kimliği,
- ayrıştırılabilir ilan tarihi,
- sıfırdan büyük fiyat,
- sıfırdan büyük brüt ve net alan.

Bu kontroller 9 satırı ayırmıştır. Çıkarılan satırlar
`outputs/latest/tables/reddedilen_kayitlar.csv` dosyasında gerekçeleriyle korunur.

## Gözlem birimi ve sayıların yorumu

Ana modelin gözlem birimi **tekil ilan**dır. Yapısal olarak geçerli 2.272 görüntüden,
ilan numarası başına en güncel kayıt seçilerek 1.922 gözleme ulaşılır.

- `exact_duplicate_rows = 169`: Excel içeriği bakımından tam kopya satır sayısıdır.
- `repeated_snapshot_rows = 665`: Birden fazla geçerli görüntüsü bulunan ilanlara ait
  tüm satırları sayar; yalnızca elenen eski görüntülerin sayısı değildir.
- `latest_snapshot_rows = 1.922`: Ana analizdeki tekil ilan sayısıdır.
- `previously_removed_rows = 555`: Bildirilen ham 2.836 kayıt ile repodaki 2.281 kayıt
  arasındaki farktır; bu proje sırasında silinen satırlar değildir.

Bu dört değer farklı kavramları ölçer ve toplanmamalıdır.

## Eksik değer yaklaşımı

Kategorik eksikler `Belirtilmemiş` sınıfına alınır. Sayısal model girdileri eğitim
verisinin medyanıyla doldurulur ve eksiklik göstergesi eklenir. Doldurma ve kodlama
yalnızca ilgili eğitim katında öğrenildiği için test bilgisi ön işleme adımlarına sızmaz.
