# Hyperparameter Tuning Rehberi

Her algoritma için hazır GridSearchCV kodu. Kendi notebook'una kopyala-yapıştır, çalıştır.

---

## 0. Veri Yükleme — Tüm tuning notebook'larının başına ekle

Preprocessing zaten yapıldı (`02_preprocessing.ipynb`). Final veriyi doğrudan yükle:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

RANDOM_STATE = 42

X_train = pd.read_csv('../data/X_train_final.csv')
X_test  = pd.read_csv('../data/X_test_final.csv')
y_train = pd.read_csv('../data/y_train_final.csv').iloc[:, 0]
y_test  = pd.read_csv('../data/y_test_final.csv').iloc[:, 0]

print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

# Yardımcı: model değerlendirme
def evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n=== {name} ===')
    print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
    print(f'F1 macro : {f1_score(y_test, y_pred, average="macro"):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred, average="macro"):.4f}')
    print(f'Recall   : {recall_score(y_test, y_pred, average="macro"):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    return model
```

> **Not:** Veri zaten scaled durumda (RobustScaler), encoding tamamlanmış (OHE + Target Encoding), feature engineering yapılmış (arac_yasi, km_per_yas, vb.). Tekrar yapma!

---

## k-NN

```python
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"En iyi parametreler: {grid.best_params_}")
print(f"En iyi CV skoru    : {grid.best_score_:.4f}")

best_knn = grid.best_estimator_
evaluate(best_knn, X_train, X_test, y_train, y_test, 'k-NN (optimized)')
```

**Süre:** ~1-2 dakika.

---

## ID3 (entropy-based karar ağacı)

```python
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['entropy'],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"En iyi parametreler: {grid.best_params_}")
best_id3 = grid.best_estimator_
evaluate(best_id3, X_train, X_test, y_train, y_test, 'ID3 (optimized)')
```

**Süre:** ~1 dakika.

---

## C4.5 (entropy + pruning)

```python
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['entropy'],
    'max_depth': [5, 8, 10, 15],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.05],  # Cost complexity pruning
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"En iyi parametreler: {grid.best_params_}")
best_c45 = grid.best_estimator_
evaluate(best_c45, X_train, X_test, y_train, y_test, 'C4.5 (optimized)')
```

**Not:** `ccp_alpha` budama parametresi — C4.5'in ayırt edici özelliği.

---

## CART (gini-based)

```python
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_cart = grid.best_estimator_
evaluate(best_cart, X_train, X_test, y_train, y_test, 'CART (optimized)')
```

---

## Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

param_grid = {
    'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
}

grid = GridSearchCV(
    GaussianNB(),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_nb = grid.best_estimator_
evaluate(best_nb, X_train, X_test, y_train, y_test, 'Naive Bayes (optimized)')
```

**Not:** Naive Bayes'in çok az parametresi var, büyük iyileşme beklenmez.
Raporunda "bu veri için neden uygun değil" kısmını vurgula:
- Feature'lar arasında bağımsızlık varsayımı geçerli değil (motor gücü ↔ motor hacmi gibi)
- Sürekli değişkenler için Gaussian varsayımı zayıf

---

## Lojistik Regresyon

```python
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [2000],
}

grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_lr = grid.best_estimator_
evaluate(best_lr, X_train, X_test, y_train, y_test, 'Lojistik Regresyon (optimized)')
```

**Süre:** ~3-5 dakika.

---

## K-Means (kümelerle sınıflandırma)

K-Means bir sınıflandırma algoritması değil, bu yüzden farklı bir yaklaşım:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method ile optimal küme sayısı
inertias = []
silhouette_scores = []
k_values = [2, 3, 4, 5, 6, 7, 8]

for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_train)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_train, labels))

# Elbow grafiği
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(k_values, inertias, 'o-')
axes[0].set_xlabel('Küme sayısı (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

axes[1].plot(k_values, silhouette_scores, 'o-')
axes[1].set_xlabel('Küme sayısı (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# En iyi k ile sınıflandırma
best_k = 4  # Grafiğe göre seç
km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
train_clusters = km.fit_predict(X_train)

cluster_to_label = {}
for c in range(best_k):
    mask = train_clusters == c
    if mask.sum() > 0:
        cluster_to_label[c] = y_train[mask].mode()[0]

test_clusters = km.predict(X_test)
y_pred_km = [cluster_to_label[c] for c in test_clusters]

print(f"Accuracy: {accuracy_score(y_test, y_pred_km):.4f}")
print(f"F1 macro: {f1_score(y_test, y_pred_km, average='macro'):.4f}")
```

**Raporda:** K-Means'ın düşük skoruna takılma. Rapor etmen gerekenler:
- Unsupervised, sınıflandırma için özel tasarlanmamış
- Elbow ve silhouette analizi yaptın
- Cluster-based feature ekleyerek başka bir sınıflandırıcıya destek verilebilir

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Çok kombinasyon var — RandomizedSearchCV kullan
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid,
    n_iter=30,  # 30 rastgele kombinasyon dene
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
evaluate(best_rf, X_train, X_test, y_train, y_test, 'Random Forest (optimized)')
```

**Süre:** ~5-10 dakika. Bonus: feature importance'ı gör:

```python
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
plt.title('En önemli 15 özellik')
plt.tight_layout()
plt.show()
```

---

## Confusion Matrix — herkes için zorunlu

Raporunda hangi sınıfları karıştırdığını göstermek için:

```python
sinif_sirasi = ['Ekonomik', 'Orta', 'Yüksek', 'Premium']

cm = confusion_matrix(y_test, best_model.predict(X_test), labels=sinif_sirasi)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sinif_sirasi)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix — [Senin Algoritman]')
plt.tight_layout()
plt.savefig('../outputs/confusion_matrix_[algoritma].png', dpi=150)
plt.show()
```

---

## Notlar

- Tüm tuning notebook'larında `random_state=42` kullan — sonuçların karşılaştırılabilir olması için.
- `cv=5` (5-fold cross-validation) standart — değiştirme.
- Veri zaten preprocessed; ek scaling, encoding, FE yapma.
- Pipeline'ı bozmaman önemli: tüm üyeler aynı `X_train_final.csv` ile çalışıyor.
