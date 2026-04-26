# Ekip Üyesi Başlangıç Rehberi

## 1. Repo'yu bilgisayarına al

```bash
git clone <repo_url>
cd veri-madenciligi-proje
```

## 2. Veri dosyasını al

`ARABAMVS.csv` dosyasını liderden iste (WhatsApp/Drive). Bu dosyayı `data/` klasörünün içine koy.

⚠️ **Veri dosyalarını repoya COMMIT ETME** — `.gitignore`'da `data/*.csv` zaten engelli.

## 3. Gerekli kütüphaneleri kur

### Seçenek A — Anaconda kullananlar

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

### Seçenek C — Doğrudan pip

```bash
pip install pandas numpy scikit-learn matplotlib seaborn missingno openpyxl jupyter scipy
```

## 4. Önce verinin hazırlanması (1 kez yap)

### 4.1 EDA (`notebooks/01_EDA.ipynb`) — opsiyonel ama tavsiye edilir

Keşifsel analiz — veri yapısı, eksik değerler, dağılımlar, korelasyonlar.

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

**Kernel → Restart Kernel and Run All Cells** ile baştan çalıştır. 1-2 dakika sürer. Veri hakkında bilgin olur, modelleme yorumu yazarken kullanırsın.

### 4.2 Preprocessing (`notebooks/02_preprocessing.ipynb`) — ZORUNLU

10 adımlı pipeline. Çıktı: 4 final CSV dosyası (modelleme bunları okuyacak).

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**Kernel → Restart Kernel and Run All Cells**

3-5 dakika sürer. Bittikten sonra `data/` klasöründe 4 final dosya oluşmuş olmalı:

```
data/X_train_final.csv  →  2739 × 74
data/X_test_final.csv   →   685 × 74
data/y_train_final.csv  →  2739 etiket
data/y_test_final.csv   →   685 etiket
```

⚠️ **Bu adımı tekrar yapma sürekli** — bir kez çalıştır, dosyalar oluşunca yeter. Preprocessing herkes için aynı (karşılaştırılabilirlik şart).

## 5. Kendi algoritmanı geliştir (asıl iş)

### 5.1 Şablonu kopyala

`notebooks/03_template.ipynb` dosyasını kopyala, kendi algoritmana göre yeniden adlandır:

```bash
cp notebooks/03_template.ipynb notebooks/03_knn.ipynb
# veya
cp notebooks/03_template.ipynb notebooks/03_naive_bayes.ipynb
# vb. — sana atanan algoritmaya göre
```

### 5.2 Şablonu doldur

Kopyaladığın notebook'ta 5 bölüm var:

| Bölüm | İçerik | Senin işin |
|---|---|---|
| 1 — Veri Yükleme | Hazır kod | Hiçbir şey değiştirme |
| 2 — Baseline | `>>> SENİN ALGORİTMAN <<<` etiketleri | Kendi algoritmanın import + sınıfını yaz |
| 3 — Tuning | Boş `param_grid` | `docs/TUNING_REHBERI.md`'den hangi parametreleri tune edeceğini öğren, **değerleri kendin araştır** (sklearn dokümanı + ders notları) |
| 4 — Karşılaştırma | Hazır kod (tablo, bar chart, confusion matrix) | Hiçbir şey değiştirme — otomatik çalışır |
| 5 — Yorum | 4 alt bölüm soruları (markdown) | Kendi cümlelerinle doldur |

### 5.3 Çalıştır

**Kernel → Restart Kernel and Run All Cells** — hata olmadığından emin ol. Tuning bölümü algoritmaya göre 1-10 dakika sürebilir.

### 5.4 Önemli kurallar

- ⚠️ **Preprocessing'i tekrar yapma** — `data/X_train_final.csv` dosyasından doğrudan oku
- ⚠️ **`random_state=42`** — değiştirme, sonuçların liderle bire bir uyuşması gerek
- ⚠️ **Kendi algoritmanı tek dosyada bitir** — baseline + tuning + yorum hepsi `03_[algoritma].ipynb`'da
- ⚠️ Bölüm 5'teki yorumları **kendi cümlelerinle** yaz (rapor için lazım)

## 6. Git kullanımı

**Kendi branch'inde çalış, main'e direkt push ETME:**

```bash
# Kendi branch'ini aç (algoritmanın adıyla)
git checkout -b feature/knn

# Notebook'ta değişiklik yap, kaydet
git add notebooks/03_knn.ipynb
git commit -m "k-NN baseline + tuning eklendi, F1: 0.82"
git push origin feature/knn
```

GitHub'da **Pull Request (PR)** aç → liderin onaylamasını bekle.



