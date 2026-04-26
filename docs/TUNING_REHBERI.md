# Hyperparameter Tuning Rehberi

> Bu rehber **hangi parametreleri** tune etmen gerektiğini söyler. **Değerleri kendin araştır** — sklearn dokümanı, ders slaytları, akademik makaleler kullan. Raporda "neden bu değerleri seçtim?" sorusuna cevabın olmalı.
>
> Genel kalıp:
> ```python
> param_grid = {
>     'parametre_1': [...],   # ← Sen doldur, gerekçesini raporda anlat
>     'parametre_2': [...],
> }
> grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
> ```

---

## k-NN (`KNeighborsClassifier`)

**Algoritma:** Yeni veri için en yakın **k komşunun çoğunluk sınıfını** seçer.

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `n_neighbors` | Kaç komşuya bakılsın? | Küçük k → varyans yüksek (overfit), büyük k → bias yüksek. Tek sayılar tercih edilir (eşitlik kırmak için). |
| `weights` | Komşuların oyu eşit mi? | `'uniform'` her komşu eşit oy, `'distance'` yakın komşu daha çok ağırlık. |
| `metric` | Mesafe nasıl hesaplanır? | `'euclidean'` klasik, `'manhattan'` aykırı değerlere dayanıklı, `'minkowski'` p parametreli. |

**Notlar:**
- Veri zaten scaled — mesafe hesabı dengeli çalışır.
- Tek sayı k'ler dene (3, 5, 7 gibi).

---

## ID3 (`DecisionTreeClassifier(criterion='entropy')`)

**Algoritma:** Information Gain'e göre en bilgi verici özelliği seç, ona göre dalla. Entropy minimize edilir.

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `criterion` | Bölme kriteri | ID3 için **`'entropy'`** sabit, başka değer girme. |
| `max_depth` | Ağacın maksimum derinliği | Küçük → underfit, büyük → overfit. `None` sınırsız. |
| `min_samples_split` | Bir düğümü bölmek için min örnek sayısı | Büyük değer → daha az detay, daha genel ağaç. |
| `min_samples_leaf` | Yaprak düğümde min örnek | Çok küçük → overfit. |

**Notlar:**
- `random_state=RANDOM_STATE` her zaman ekle.
- `max_depth=None` ile başlamak ne kadar derinleşeceğini gösterir, sonra sınırla.

---

## C4.5 (`DecisionTreeClassifier(criterion='entropy', ccp_alpha=...)`)

**Algoritma:** ID3'ün gelişmiş hali — entropy + **cost complexity pruning** (sonradan budama).

### Tune edilebilir parametreler

C4.5 ID3'ün tüm parametrelerine sahip + ek olarak:

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `ccp_alpha` | Budama agresifliği | 0 → budama yok (= ID3), büyük değer → çok budama (basit ağaç). C4.5'in ayırt edici parametresi. |

**Notlar:**
- ID3 parametreleriyle (`max_depth`, `min_samples_split`, vb.) birlikte kullan.
- `ccp_alpha`'yı ufak değerlerle dene (0.0, 0.001 gibi → çok küçük ondalıklar).

---

## CART (`DecisionTreeClassifier(criterion='gini')`)

**Algoritma:** Gini impurity'e göre böler (entropy'ye matematiksel olarak benzer ama hesaplama farklı).

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `criterion` | **`'gini'`** sabit (CART tanımı bu). |
| `max_depth` | ID3 ile aynı |
| `min_samples_split` | ID3 ile aynı |
| `min_samples_leaf` | ID3 ile aynı |
| `max_features` | Her bölmede kaç özelliğe bakılsın | `None` hepsine bakar, `'sqrt'` rastgele kök kadar. Random Forest mantığı. |

---

## Naive Bayes (`GaussianNB`)

**Algoritma:** Bayes teoremi + özellikler arası bağımsızlık varsayımı. Her özellik Gaussian dağıldığını varsayar.

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `var_smoothing` | Varyansa eklenen küçük sayı (sıfıra bölünme önler) | Çok küçük değerler dene (1e-11, 1e-10, ...). Etkisi sınırlı. |

**Notlar:**
- Naive Bayes'in az parametresi var → büyük iyileşme beklenmez.
- Raporda **niye yetersiz** kaldığını anlat:
  - Özellikler arasında bağımsızlık varsayımı (örn. Motor Hacmi ↔ Silindir Sayısı bağımsız değil)
  - Scaled veride Gaussian varsayımı zayıf
- Alternatif: `CategoricalNB` denenebilir (sadece kategorik özelliklerle).

---

## Lojistik Regresyon (`LogisticRegression`)

**Algoritma:** Doğrusal sınır + softmax. Çok sınıflı için one-vs-rest veya multinomial.

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `C` | Regülarizasyon **tersi** | Küçük C → güçlü regülarizasyon (basit model), büyük C → az regülarizasyon. |
| `penalty` | Regülarizasyon tipi | `'l1'` seyrek katsayı (özellik seçimi), `'l2'` standart, `'elasticnet'` ikisinin karışımı. |
| `solver` | Optimizer | `'liblinear'` küçük veri, `'saga'` büyük veri + l1. Penalty ile uyumlu olmalı. |
| `max_iter` | Max iterasyon | Yakınsama uyarısı alırsan artır. |

**Notlar:**
- `random_state=RANDOM_STATE` ekle.
- Penalty + solver eşleşmesine dikkat (sklearn dokümanında uyumluluk tablosu var).

---

## K-Means (`KMeans`)

**Algoritma:** **Kümeleme** algoritması — sınıflandırma değil. 4 küme oluştur, her kümeyi en sık sınıfa eşle.

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `n_clusters` | Küme sayısı | 4 hedef sınıfımız var → 4 mantıklı, ama elbow & silhouette ile 2-8 arası dene. |
| `init` | Başlangıç merkezleri | `'k-means++'` akıllı seçim (default), `'random'` rastgele. |
| `n_init` | Kaç farklı başlangıçtan çalışsın | Yüksek → daha sağlam ama yavaş. |

**Notlar:**
- Cluster → label eşleştirmesi yapacaksın: her küme içinde en sık etiket o kümenin tahmini.
- Düşük F1 skoru beklenir (kümeleme zaten classification için tasarlanmamış).
- Raporda elbow + silhouette grafiklerini göster.

### Cluster → label eşleştirme örneği
```python
km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
train_clusters = km.fit_predict(X_train)

cluster_to_label = {}
for c in range(4):
    mask = train_clusters == c
    cluster_to_label[c] = y_train[mask].mode()[0]

test_clusters = km.predict(X_test)
y_pred_kmeans = [cluster_to_label[c] for c in test_clusters]
```

---

## Random Forest (`RandomForestClassifier`)

**Algoritma:** Birçok karar ağacının oy birliği. Bagging + rastgele özellik altkümesi.

### Tune edilebilir parametreler

| Parametre | Ne yapar | Düşünce yolu |
|---|---|---|
| `n_estimators` | Ağaç sayısı | Çok ağaç → daha kararlı ama yavaş. 100 standart başlangıç. |
| `max_depth` | Ağaç derinliği | Çok büyük → overfit, çok küçük → underfit. |
| `max_features` | Her bölmede kaç özelliğe bakılsın | `'sqrt'` standart, `'log2'` daha agresif. |
| `min_samples_split` | DT ile aynı |
| `min_samples_leaf` | DT ile aynı |

**Notlar:**
- Çok parametre var → **`RandomizedSearchCV`** tercih edilir (`GridSearchCV` çok yavaş).
- `n_jobs=-1` ile paralel çalışır.
- Feature importance grafiği zorunlu (raporda).

### RandomizedSearchCV kullanım kalıbı
```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=30,            # Kaç rastgele kombinasyon dene
    cv=5,
    scoring='f1_macro',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
```

---

## Genel İpuçları

1. **Önce küçük grid ile başla** (her parametre 2-3 değer) → çalıştığını gör → genişlet
2. **`grid.cv_results_`'a bak** → hangi parametre kombinasyonu fark yaratıyor?
3. **`grid.best_params_`'ı raporda yorumla** — niye optimal? Algoritma teorisinden açıkla
4. **Yakınsama uyarısı** alırsan `max_iter` veya `tol` parametresini düzenle
5. **`verbose=1`** ekle → progress görürsün, ne kadar sürdüğünü anlarsın
6. **Bilinmeyen parametreler** için: `from sklearn.X import Y; help(Y)` veya sklearn dokümanına bak

## Yardımcı Kaynaklar

### sklearn API dokümanı (her algoritmanın direkt linki)

Her sayfada **Parameters** bölümünden geçerli değerleri ve varsayılanları gör. **Examples** bölümünde çalışır kod var.

| Algoritma | Link |
|---|---|
| k-NN | https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html |
| Karar Ağaçları (ID3 / C4.5 / CART) | https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html |
| Naive Bayes (Gaussian) | https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html |
| Naive Bayes (Categorical — alternatif) | https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html |
| Lojistik Regresyon | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html |
| K-Means | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html |
| Random Forest | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html |
| GridSearchCV | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html |
| RandomizedSearchCV | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html |

### Notebook içinden alternatif

İnternet açmadan parametre listesi:
```python
from sklearn.neighbors import KNeighborsClassifier
help(KNeighborsClassifier)        # Terminale dökümanı bas
KNeighborsClassifier?             # IPython/Jupyter kısayolu (aynı şey)
```

### Diğer kaynaklar

- Ders slaytları 
- "Hands-On ML" — Aurélien Géron (üniversite kütüphanesinde olabilir)
- Stack Overflow: spesifik hata için arama yap
