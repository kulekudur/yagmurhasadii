# 🎯 Hızlı Başlangıç Kılavuzu - Geliştirilmiş Simülasyon

## Başlangıç

### 1. Uygulamayı Çalıştır
```bash
streamlit run app.py
```

### 2. Tarayıcıda Açılacak URL
```
http://localhost:8501
```

---

## 🆕 YENİ ÖZELLİKLER

### 1️⃣ Gerçekçi 3D Bina
- **Duvarlar**: Açık gri renkli, gerçekçi görünüm
- **Çatı**: Kırmızı eğimli çatı (piramidal forma)
- **Pencereler**: 3 seviye × 3 sütun, mavi çerçeveler

### 2️⃣ İnsanoid İşçi Modelleri
- **Yapı**: Baş (küre), gövde (çizgi), 4 uzuv
- **Renkler**: Ten rengi, turuncu gömlek, koyu pantolon
- **Zekâ**: Sadece çalışma saatleri (09:00-17:00) görünür

### 3️⃣ Yağış Animasyonu ⭐
- **Dinamik Düşüş**: Partiküller gerçekten yukarıdan aşağıya düşüyor
- **Yoğunluk Bağlantısı**: Yağış ne kadar çok olursa o kadar çok partiküller
- **Animasyon Düğmesi**: "▶️ Yağışı Canlandır" - Her tıklamada hareket artar

### 4️⃣ 100% Türkçe Ara Yüz
- **Başlık**: "🌧️ Yağmur Hasadı Sistemi Simülatörü"
- **Tüm Etiketler**: Düğmeler, kaydırıcılar, metrikler Türkçe
- **Sekme Adları**: 
  - 📊 Genel Bakış
  - 🎬 3D Görselleştirme
  - 📈 Grafikler
  - 💰 Ekonomi
  - 📥 Dışa Aktar

### 5️⃣ Zaman Kontrolleri
- **Gün Kaydırıcısı**: 0-364 gün seç
- **Saat Kaydırıcısı**: 0-23 saat seç
- **Canlı Metrikler**: Seçilen gün/saat için otomatik güncelleme

---

## 📋 ADIM ADIM KULLANIM

### Adım 1: Parametreleri Ayarla
1. Kenar çubuğu "⚙️ Simülasyon Parametreleri" bölümünden:
   - Çatı Alanı (m²)
   - Toplama Verimliliği (%)
   - Depo Kapasitesi (L)
   - Çalışan Sayısı
   - Tüketim Oranı
   - Yağış Tohumu

2. "💰 Ekonomik Parametreler":
   - Su Fiyatı (₺/m³)
   - Depo Maliyeti (₺)
   - Yıllık Bakım (₺)

### Adım 2: Simülasyonu Çalıştır
- Düğme: "▶️ Simülasyonu Çalıştır"
- Bekleme: ~10 saniye (365 gün simülasyon)
- Başarı mesajı görüntülenir

### Adım 3: 3D Görselleştirmede Keşfet
1. "🎬 3D Görselleştirme" sekmesine git
2. Gün ve Saat seçicilerini kullan
3. Sonuçları gözlemle:
   - Bina, depo, yağış, işçiler
4. "▶️ Yağışı Canlandır" düğmesine tıkla → Yağış hareket eder!

### Adım 4: Grafikleri İncele
1. "📈 Grafikler" sekmesine git
2. Farklı grafikleri seç:
   - 🚰 Depo Seviyesi (zaman içinde)
   - 🌧️ Yağış (günlük dağılım)
   - ⚖️ Arz vs Talep
   - 📊 Kümülatif Su

### Adım 5: Ekonomik Analizi Oku
1. "💰 Ekonomi" sekmesine git
2. Önemli metrikler:
   - Tasarruf Edilen Su (₺)
   - Toplam Yatırım (₺)
   - Net Fayda (₺)
   - ROI (%)
   - Geri Ödeme Süresi

### Adım 6: Verileri İndir
1. "📥 Dışa Aktar" sekmesine git
2. İndirme seçenekleri:
   - CSV (Günlük veriler)
   - JSON (Tüm sonuçlar)

---

## 🎨 3D SAHNESİ ANLAMAK

### Bina Bileşenleri
```
┌─────────────────┐  ← Kırmızı çatı (pik 3m yukarı)
│   PENCERELER   │  ← 9 mavi pencere
│  [□] [□] [□]   │  
│  [□] [□] [□]   │
│  [□] [□] [□]   │
└─────────────────┘  ← Gri duvarlar
```

### İşçi Modeli
```
      ⭕   ← Baş (ten rengi)
     ╱ ╲  ← Kollar (ten rengi)
    |   | ← Gövde (turuncu)
     ╲ ╱  ← Bacaklar (koyu gri)
     │ │
```

### Depo Tasarımı
```
   /¯¯¯¯¯\   ← Üst kapağı
  |  Su  |   ← Mavi: Dolu kısım
  |  Su  |   ← Gri: Boş kısım
  |XXXXXX|   ← Suyu temsil eden desen
   \____/    ← Taban plakası
```

### Yağış Animasyonu
```
Yoğun yağış:     Hafif yağış:      Yağış yok:
∴ ∴ ∴ ∴ ∴        ·  ·  ·           (boş)
 ∴ ∴ ∴ ∴          ·  ·
∴ ∴ ∴ ∴ ∴          ·
```

---

## ⏰ ÇALIŞMA SAATLERİ

### İşçiler Ne Zaman Görünür?
- ✅ **Saat 09:00 - 17:00**: İşçiler aktif ve görünür
- ❌ **Diğer saatler**: İşçiler nonaktif ve gizli

### Görsel İşaret
```
Çalışma saatleri içinde:
✓ Çalışma saatleri içinde (09:00–17:00) - İşçiler aktif
[Yeşil info kutusunda]

Dışında:
✗ Çalışma saatleri dışında - İşçiler pasif
[Sarı uyarı kutusunda]
```

---

## 🔍 METRİKLERİ OKUMA

### 🏠 Genel Bakış Sekmesi
| Kısaltma | Anlamı | Normal Aralık |
|----------|--------|----------------|
| Toplam Toplanan Su | 365 gün boyunca çatıdan toplanan | >100.000 L |
| Toplam Tüketilen Su | Depodan kullanılan | <100.000 L |
| Yetersiz Gün | Su eksikliği olan günler | 0-50 gün |
| Depo Doluluk % | Ortalama depo seviyesi | 20-80% |

### 3D Görselleştirme
| Metrik | Açıklama |
|--------|----------|
| Gün | Seçilen simülasyon günü (0-365) |
| Saat | Seçilen saat (0-23) |
| Depo Doluluk % | O anki depo doldurma yüzdesi |
| Yağış | O gün yağış miktarı (mm) |

### 💰 Ekonomik Analiz
| Terim | Anlamı |
|-------|--------|
| Tasarruf Edilen Su | Yağmurdan toplanan su satın almaktan tasarruf |
| Toplam Yatırım | Depo + kurulum + bakım |
| ROI | Yatırım getirisi yüzdesi |
| Geri Ödeme Süresi | Yatırımı geri kazanma zamanı |
| Ekonomik Durumda Uygun | 20 yıl içinde kârlı mı? |

---

## 🚨 SORUN GİDERME

### Problem: Yağış görmüyorum
**Çözüm**: 
- Yağış şiddeti çok düşük → Gün değiştir
- "▶️ Yağışı Canlandır" düğmesine tıkla → Animasyon başlar

### Problem: İşçiler görünmüyor
**Çözüm**:
- Saat 9-17 arasında mı? Değilse sadece bu saatlerde görünür
- Saat kaydırıcısını 12'ye ayarla (öğlen)

### Problem: 3D sahne yavaş
**Çözüm**:
- Tarayıcıyı yenile (F5)
- Streamlit uygulamasını yeniden başlat
- İnternet bağlantısı kontrol et

### Problem: Türkçe karakterler garbled
**Çözüm**:
- Dosya kodlaması: UTF-8 olmalı
- Tarayıcı ayarları: UTF-8 kullan

---

## 📊 VERİ DIŞA AKTARMASI

### CSV Formatı
Günlük veri indirme (365 satır):
```
day,rainfall,collected,consumed,tank_level,water_shortage
0,2.5,2125.0,2400.0,99600,False
1,0.0,0.0,2400.0,97200,False
...
```

### JSON Formatı
Tüm simülasyon sonuçları (metrikleri, grafikleri, tüm veriler):
```json
{
  "water_metrics": {...},
  "tank_metrics": {...},
  "economic_metrics": {...},
  "daily_history": [...],
  ...
}
```

---

## 💡 İPUÇLARI

✨ **Maksimum Deneyim İçin**:
1. Gün 150-200'ü seçerek ilk sonuçları gözlemle
2. Saat 12'yi seçerek işçileri görmek mümkün
3. Yağışlı günleri bul (yağış yoğunluğu >20mm)
4. "Yağışı Canlandır" düğmesine birkaç kez tıkla
5. Grafikleri tüm yıl için incele

🎯 **Ekonomik Analiz İçin**:
- Çatı alanını 1500 m² yaparak test et
- Depo maliyetini değiştir ve ROI değişimini göz
- "Geri Ödeme Süresi" düşük olması iyi demek

---

## 📞 DESTEK

Sorularınız için:
- Kod: `modules/visualization.py` ve `app.py` inceleyin
- Yapılandırma: `config.py` parametreleri değiştirin
- Doküman: `IMPROVEMENTS_SUMMARY.md` gibi belgeler

---

🌧️ **Rahat simülasyon keyfi dileriz!** ✨
