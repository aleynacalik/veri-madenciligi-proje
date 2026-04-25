# Ekip Üyesi Başlangıç Rehberi

## 1. Repo'yu bilgisayarına al

```bash
git clone <repo_url>
cd veri-madenciligi-proje
```

## 2. Veri dosyasını al

`ARABAMVS.csv` dosyasını al.
Bu dosyayı `data/` klasörünün içine koy.



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

## 4. Notebook'ları sırayla çalıştır

### 4.1 EDA (`notebooks/01_EDA.ipynb`)

Keşifsel analiz — veri yapısı, eksik değerler, dağılımlar, korelasyonlar.

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

**Kernel → Restart Kernel and Run All Cells** ile baştan çalıştır.

### 4.2 Preprocessing (`notebooks/02_preprocessing.ipynb`)

10 adımlı pipeline. Çıktı: `data/X_train_final.csv`, `X_test_final.csv`, `y_train_final.csv`, `y_test_final.csv`.

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

### 4.3 Baseline (`notebooks/03_baseline.ipynb`)

Henüz oluşturulmadı — proje takvimine göre eklenecek.

## 5. Kendi algoritmanı optimize et

Sana atanan algoritmayı `04_tuning.ipynb` (eklenecek) içinde optimize et. Hazır GridSearchCV kodları için → `docs/TUNING_REHBERI.md`.

**Önemli:** Modelleme notebook'ların `data/X_train_final.csv` ve `data/X_test_final.csv` dosyalarından okumalı. Preprocessing'i tekrar yapma — herkes aynı işlenmiş veriyi kullanıyor (karşılaştırılabilirlik için).

## 6. Git kullanımı

**Kendi branch'inde çalış, main'e direkt push ETME:**

```bash
# Kendi branch'ini aç (algoritmanın adıyla)
git checkout -b feature/knn-tuning

# Notebook'ta değişiklik yap, kaydet
git add notebooks/04_tuning_knn.ipynb
git commit -m "k-NN GridSearch eklendi, F1: 0.82"
git push origin feature/knn-tuning
```

GitHub'da Pull Request (PR) aç, lider onaylasın.


