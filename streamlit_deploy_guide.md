# Streamlit Cloud'a Deploy Rehberi

## Adım 1: Gerekli Dosyalar
✅ `requirements.txt` - Hazır
✅ `.streamlit/config.toml` - Hazır
✅ `.gitignore` - Hazır
✅ `app.py` - Mevcut

## Adım 2: GitHub'a Push Edin

```bash
# Git repository oluşturun (eğer yoksa)
git init

# Dosyaları stage edin
git add .

# Commit yapın
git commit -m "Rainwater Harvesting Simulator v2.0"

# Remote ekleyin
git remote add origin https://github.com/YOUR_USERNAME/rainwater-harvesting.git

# Main branch'e push edin
git branch -M main
git push -u origin main
```

## Adım 3: Streamlit Cloud'a Bağlanın

1. [Streamlit Cloud](https://streamlit.io/cloud) adresine gidin
2. GitHub hesabınızla giriş yapın
3. "New app" butonuna tıklayın
4. Aşağıdaki bilgileri girin:
   - **Repository**: YOUR_USERNAME/rainwater-harvesting
   - **Branch**: main
   - **Main file path**: app.py

5. "Deploy" butonuna tıklayın

## Adım 4: Ortam Değişkenlerini Ayarlayın (Eğer gerekiyorsa)

1. Deploy ettikten sonra app ayarlarına gidin
2. "Advanced settings" > "Secrets" kısmına gidin
3. Gerekli ortam değişkenlerini ekleyin:

```toml
# .streamlit/secrets.toml
DATABASE_URL="postgresql://..."
REDIS_URL="redis://..."
```

## Adım 5: Uygulamayı Test Edin

Deploy tamamlandıktan sonra:
- Streamlit tarafından sağlanan URL'yi ziyaret edin
- Uygulamanın doğru çalıştığını kontrol edin
- Simülasyonu çalıştırın

## Sorun Giderme

### Port Hatası
Eğer port hatası alırsanız, `config.toml` dosyasında port 8501 olarak ayarlı olduğundan emin olun.

### Bağımlılık Hatası
requirements.txt dosyasının güncellenmiş olduğundan emin olun:
```bash
pip install -r requirements.txt
```

### Veri Dosyaları
Eğer uygulamanız veri dosyaları kullanıyorsa, bunları GitHub'a commit etmeyi veya Streamlit Secrets kullanarak harici kaynaklara bağlamayı düşünün.

## Lokal Test (Deploy Öncesi)

```bash
# Virtual environment aktivasyon (Windows)
.\venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı çalıştır
streamlit run app.py
```

Uygulamanız `http://localhost:8501` adresinde açılmalıdır.
