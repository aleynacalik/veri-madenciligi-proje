# Veri Madenciliği Dönem Projesi

**Ders:** ISE 302 Veri Madenciliği — Sakarya Üniversitesi
**Konu:** İkinci El SUV Fiyat Sınıflandırma
**Veri kaynağı:** arabam.com (Nissan + Hyundai SUV, 3424 ilan)

---

## 🎯 Problem

İkinci el SUV ilanlarının özelliklerinden (yıl, km, motor gücü, marka vs.) **fiyat sınıfını** tahmin etmek.

- **Problem tipi:** Çok sınıflı sınıflandırma
- **Hedef:** `Fiyat_Sinifi` — 4 kategori (Ekonomik / Orta / Yüksek / Premium)

## 📊 Baseline Sonuçlar

Hyperparameter tuning ÖNCESİ:

| Sıra | Model | F1 (macro) |
|------|-------|-----------|
| 🥇 | Random Forest | 0.8649 |
| 🥈 | CART | 0.8465 |
| 🥉 | ID3 | 0.8304 |
| 4 | Lojistik Regresyon | 0.7919 |
| 5 | C4.5 | 0.7905 |
| 6 | k-NN | 0.7739 |
| 7 | K-Means | 0.3582 |
| 8 | Naive Bayes | 0.3247 |

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
├── PROJE_TEK_DOSYA.ipynb   # Ana notebook — baseline tüm algoritmalar
├── data/                    # Veri dosyaları (.gitignore'da)
│   └── arabam_ham_veri.xlsx
├── docs/                    # Dokümantasyon
│   ├── BASLANGIC.md         # Ekip üyesi kurulum rehberi
│   ├── TUNING_REHBERI.md    # Her algoritma için tuning kodu
│   └── proje_isterleri.pdf  # Hocanın verdiği proje isteri
├── outputs/                 # Çıktılar
│   ├── sonuclar.csv
│   ├── arabam_temiz.csv
│   └── model_karsilastirma.png
└── README.md
```

## 🚀 Hızlı Başlangıç

```bash
# 1. Klonla
git clone <repo_url>
cd veri-madenciligi-proje

# 2. Ham veriyi `data/` klasörüne koy
# (dosya adı: arabam_ham_veri.xlsx)

# 3. Notebook'u aç
jupyter notebook PROJE_TEK_DOSYA.ipynb

# 4. Kernel → Restart & Run All
```

Detaylı kurulum için → `docs/BASLANGIC.md`

## 🗓️ Takvim

| Hafta | Aşama | Durum |
|-------|-------|-------|
| 1-4 | Veri toplama (ORTAK) | ✅ |
| 4-5 | Ön işleme | ✅ |
| 6-9 | Her üye kendi modelini optimize eder | 🔄 |
| 10 | Hyperparameter tuning | ⏳ |
| 11-12 | Sonuç karşılaştırma + rapor | ⏳ |
| 13-14 | Sunum (15 dk) | ⏳ |

## ⚠️ Kurallar

1. **REKABET:** Grup içi sonuçlar başka gruplarla paylaşılmayacak (PDF'te yazıyor).
2. **AYNI SPLIT:** Herkes `RANDOM_STATE=42`, `test_size=0.20` kullanıyor — sonuçlar karşılaştırılabilir.
3. **GIT FLOW:** Kendi branch'inde çalış, PR açarak main'e gönder.

---

*Son güncelleme: [Tarih]*
