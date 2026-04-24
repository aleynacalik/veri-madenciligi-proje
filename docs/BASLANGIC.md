# Ekip Üyesi Başlangıç Rehberi

## 1. Repo'yu bilgisayarına al

```bash
git clone <repo_url>
cd veri-madenciligi-proje
```

## 2. Ham veri dosyasını al

Proje liderinden `arabam_ham_veri.xlsx` dosyasını al (WhatsApp/Drive).
Bu dosyayı `data/` klasörünün içine koy.

```bash
# Örnek — senin dizininde dosya nereden gelirse
cp ~/Downloads/arabam_ham_veri.xlsx data/
```

⚠️ **Ham veriyi repoya COMMIT ETME** — `.gitignore`'da zaten engelli.

## 3. Gerekli kütüphaneleri kur

### Seçenek A — Anaconda kullananlar (tavsiye edilen)

Anaconda yüklüyse büyük ihtimalle tüm kütüphaneler zaten var. Yine de emin olmak için:

```bash
pip install -r requirements.txt
```

### Seçenek B — Sanal ortam (venv) ile

Anaconda yoksa temiz bir sanal ortam kurmak daha iyi:

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Seçenek C — Doğrudan pip (hızlı ama karışık olabilir)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter scipy
```

Açılan notebook'ta üst menüden:
**Kernel → Restart Kernel and Run All Cells**

1-2 dakika sürer. Sonunda şunları görmelisin:
- 8 modelin F1 skorları
- Bir karşılaştırma grafiği
- Yeni dosyalar: `sonuclar.csv`, `arabam_temiz.csv`, `model_karsilastirma.png`

## 5. Kendi algoritmanı optimize et

Sana verilen algoritmanın hücresini bul (6.1, 6.2... gibi). O hücreye **GridSearchCV** ekle.

Her algoritma için hazır tuning kodunu `docs/TUNING_REHBERI.md` dosyasında bul.

## 6. Git kullanımı

**Kendi branch'inde çalış, main'e direkt push ETME:**

```bash
# Kendi branch'ini aç (algoritmanın adıyla)
git checkout -b feature/knn-tuning

# Notebook'ta değişiklik yap, kaydet
# Sonra:
git add PROJE_TEK_DOSYA.ipynb
git commit -m "k-NN GridSearch eklendi, F1: 0.82"
git push origin feature/knn-tuning
```

GitHub'da Pull Request (PR) aç, lider onaylasın.

## Sık sorulan sorular

**S: Jupyter açılınca "No such file or directory" hatası alıyorum**
C: Jupyter'ı `veri-madenciligi-proje` klasörünün İÇİNDEN başlat. Yani `cd veri-madenciligi-proje` yaptıktan sonra `jupyter notebook`.

**S: Baseline skorlarım tabloyla aynı mı çıkmalı?**
C: Evet, tam olarak aynı — `RANDOM_STATE=42` sağ olsun. Farklı çıkıyorsa kodu değiştirmişsindir.

**S: Benim algoritma zaten yüksek skor aldı, tuning gerekli mi?**
C: Evet. Proje isteri "optimizasyon" istiyor, raporda mutlaka GridSearch adımını göstermelisin.

**S: Naive Bayes %32 çıktı, nasıl yükseltirim?**
C: Çok yükseltemezsin — algoritma bu veriye uymuyor. Raporda "neden uymadı" analizi yap, tam not alırsın. Alternatif: `CategoricalNB` dene.
