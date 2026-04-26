# Veri Madenciliği Dönem Projesi

**Ders:** ISE 302 Veri Madenciliği — Sakarya Üniversitesi
**Konu:** İkinci El SUV Fiyat Sınıflandırma
**Veri kaynağı:** arabam.com (Nissan + Hyundai SUV, 3424 ilan)

---

## 🎯 Problem

İkinci el SUV ilanlarının özelliklerinden (yıl, km, motor gücü, marka vs.) **fiyat sınıfını** tahmin etmek.

- **Problem tipi:** Çok sınıflı sınıflandırma
- **Hedef:** `Fiyat_Sinifi` — 4 kategori (Ekonomik / Orta / Yüksek / Premium), `pd.qcut(q=4)` ile quartile tabanlı

## 📂 Notebook Yapısı

| Notebook | Durum | İçerik |
|---|---|---|
| `notebooks/01_EDA.ipynb` | ✅ | Keşifsel veri analizi (9 bölüm) |
| `notebooks/02_preprocessing.ipynb` | ✅ | 10 adımlı preprocessing pipeline |
| `notebooks/03_template.ipynb` | ✅ | Üye baseline + tuning şablonu |
| `notebooks/03_*.ipynb` × 8 | ⏳ | Her üye kendi algoritması (template'i kopyalar, doldurur) |
| `notebooks/04_comparison.ipynb` | ⏳ | Final sonuç karşılaştırması + rapor |

## 🔧 Preprocessing Adımları (02_preprocessing.ipynb)

| Adım | İşlem | Çıkış boyutu |
|---|---|---|
| 1 | %45+ eksik 3 sütun atıldı (Şanzıman, Ağır Hasarlı, Ağırlık) | 3424×40 |
| 2 | Eksiklik deseni analizi (görselleştirme) | — |
| 3 | Numerik string parse + `_Başlık` drop | 3424×39 |
| 4 | Hedef oluştur (`Fiyat_Sinifi`) | 3424×40 |
| 5 | Train/test split (80/20, stratify) | 2739 / 685 |
| 6 | Eksik doldurma (Marka+Seri grup medyanı, train-fit) | 0 eksik |
| 7 | Winsorize (`Ortalama Kasko` üst/alt %1) | 1 hatalı kırpıldı |
| 8 | Encoding + Feature Engineering (OHE, Target Encoding, türev özellikler) | 2739×74 |
| 9 | Scaling (RobustScaler, train-fit) | aynı boyut |
| 10 | Final kayıt | `data/X_*_final.csv`, `y_*_final.csv` |

### Veri Sızıntısı Önleme
Tüm istatistikler (medyanlar, modlar, scaler parametreleri, target encoding değerleri) **sadece X_train**'den hesaplandı, X_test'e aynı parametrelerle uygulandı.

## 📊 Baseline Sonuçlar

> Her üye kendi algoritmasını çalıştırdıktan sonra bu tablo doldurulacak. Hedef: F1-macro üzerinden sıralama.

| Sıra | Model | F1 (macro) |
|------|-------|-----------|
| ? | Random Forest | — |
| ? | CART | — |
| ? | ID3 | — |
| ? | C4.5 | — |
| ? | Lojistik Regresyon | — |
| ? | k-NN | — |
| ? | Naive Bayes | — |
| ? | K-Means | — |

## 👥 Ekip ve Algoritma Dağılımı

| # | Üye | Algoritma |
|---|-----|-----------|
| 1 | [İsim] — Lider | Random Forest |
| 2 | [İsim] | CART |
| 3 | [İsim] | ID3 |
| 4 | [İsim] | Lojistik Regresyon |
| 5 | [İsim] | C4.5 |
| 6 | [İsim] | k-NN |
| 7 | [İsim] | K-Means |
| 8 | [İsim] | Naive Bayes |

## 📁 Klasör Yapısı

```
veri-madenciligi-proje/
├── notebooks/
│   ├── 01_EDA.ipynb               # Keşifsel veri analizi
│   ├── 02_preprocessing.ipynb     # 10-adımlı ön işleme
│   ├── 03_template.ipynb          # Üye şablonu (kopyala → doldur)
│   └── 03_[algoritma].ipynb       # Her üye kendi algoritması (×8)
├── data/                          # .gitignore'da — repoya commit edilmez
│   ├── ARABAMVS.csv               # Ham veri (ARABAMVS sheet)
│   ├── X_train_final.csv          # Modellemeye hazır train özellikleri
│   ├── X_test_final.csv           # Modellemeye hazır test özellikleri
│   ├── y_train_final.csv          # Train hedef
│   ├── y_test_final.csv           # Test hedef
│   └── *_step{1..8}.csv           # Ara kayıtlar (her adım sonrası)
├── docs/
│   ├── BASLANGIC.md               # Ekip üyesi kurulum rehberi
│   ├── TUNING_REHBERI.md          # Algoritma tuning kodları
│   └── ders_materyalleri/         # Hafta hafta ders slaytları
├── outputs/                       # Çıktı dosyaları (grafikler, sonuç tabloları)
├── requirements.txt
└── README.md
```

## 🚀 Hızlı Başlangıç

```bash
# 1. Klonla
git clone <repo_url>
cd veri-madenciligi-proje

# 2. Veri dosyasını al
# data/ARABAMVS.csv — proje liderinden iste (WhatsApp/Drive)

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. Notebook'ları sırayla çalıştır
jupyter notebook notebooks/01_EDA.ipynb         # önce keşif
jupyter notebook notebooks/02_preprocessing.ipynb # sonra ön işleme
# (baseline ve tuning notebook'ları ekleneceğinde)
```

Detaylı kurulum için → `docs/BASLANGIC.md`

## 🗓️ Takvim

| Hafta | Aşama | Durum |
|-------|-------|-------|
| 1-4 | Veri toplama (ORTAK) | ✅ |
| 4-5 | EDA + Ön işleme pipeline | ✅ |
| 6-9 | Baseline + her üye kendi modelini optimize eder | 🔄 |
| 10 | Hyperparameter tuning | ⏳ |
| 11-12 | Sonuç karşılaştırma + rapor | ⏳ |
| 13-14 | Sunum (15 dk) | ⏳ |

## ⚠️ Kurallar

1. **REKABET:** Grup içi sonuçlar başka gruplarla paylaşılmayacak (proje isteri PDF'inde yazıyor).
2. **AYNI VERİ:** Herkes `data/X_train_final.csv` ve `data/X_test_final.csv` ile çalışıyor — sonuçlar bire bir karşılaştırılabilir.
3. **AYNI SPLIT:** `RANDOM_STATE=42`, `test_size=0.20`, `stratify=Fiyat_Sinifi`.
4. **GIT FLOW:** Kendi branch'inde çalış, PR açarak main'e gönder.

---

*Son güncelleme: 2026-04-25 — Preprocessing pipeline tamamlandı (02_preprocessing.ipynb)*
