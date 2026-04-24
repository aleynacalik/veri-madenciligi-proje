# Hyperparameter Tuning Rehberi

Her algoritma için hazır GridSearchCV kodu. Kendi hücrene kopyala-yapıştır, çalıştır.

**Not:** Hepsi `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test` değişkenlerini kullanıyor — bunlar notebook'un 4. adımında zaten oluşturuluyor.

---

## k-NN

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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
grid.fit(X_train_scaled, y_train)

print(f"En iyi parametreler: {grid.best_params_}")
print(f"En iyi CV skoru: {grid.best_score_:.4f}")

best_knn = grid.best_estimator_
evaluate(best_knn, X_train_scaled, X_test_scaled, y_train, y_test, 'k-NN (optimized)')
```

**Süre:** ~1-2 dakika. **Beklenen iyileşme:** 0.77 → 0.82

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
grid.fit(X_train_scaled, y_train)

print(f"En iyi parametreler: {grid.best_params_}")
best_id3 = grid.best_estimator_
evaluate(best_id3, X_train_scaled, X_test_scaled, y_train, y_test, 'ID3 (optimized)')
```

**Süre:** ~1 dakika. **Beklenen iyileşme:** 0.83 → 0.86

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
grid.fit(X_train_scaled, y_train)

print(f"En iyi parametreler: {grid.best_params_}")
best_c45 = grid.best_estimator_
evaluate(best_c45, X_train_scaled, X_test_scaled, y_train, y_test, 'C4.5 (optimized)')
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
grid.fit(X_train_scaled, y_train)

best_cart = grid.best_estimator_
evaluate(best_cart, X_train_scaled, X_test_scaled, y_train, y_test, 'CART (optimized)')
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
grid.fit(X_train_scaled, y_train)

best_nb = grid.best_estimator_
evaluate(best_nb, X_train_scaled, X_test_scaled, y_train, y_test, 'Naive Bayes (optimized)')
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
grid.fit(X_train_scaled, y_train)

best_lr = grid.best_estimator_
evaluate(best_lr, X_train_scaled, X_test_scaled, y_train, y_test, 'Lojistik Regresyon (optimized)')
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
    labels = km.fit_predict(X_train_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_train_scaled, labels))

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
train_clusters = km.fit_predict(X_train_scaled)

cluster_to_label = {}
for c in range(best_k):
    mask = train_clusters == c
    if mask.sum() > 0:
        cluster_to_label[c] = y_train[mask].mode()[0]

test_clusters = km.predict(X_test_scaled)
y_pred_km = [cluster_to_label[c] for c in test_clusters]

from sklearn.metrics import accuracy_score, f1_score
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
from sklearn.model_selection import RandomizedSearchCV

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
random_search.fit(X_train_scaled, y_train)

best_rf = random_search.best_estimator_
evaluate(best_rf, X_train_scaled, X_test_scaled, y_train, y_test, 'Random Forest (optimized)')
```

**Süre:** ~5-10 dakika. Bonus: feature importance'ı gör:

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
plt.title('En önemli 15 özellik')
plt.tight_layout()
plt.show()
```

---

## Herkese özellik mühendisliği önerisi

Algoritmandan önce veya sonra yeni feature'lar türetirsen skor yükselir:

```python
# Bu hücreyi 4. adımdan önce ekle
df_encoded['arac_yasi'] = 2026 - df_encoded['Yıl']
df_encoded['km_per_yil'] = df_encoded['Kilometre'] / (df_encoded['arac_yasi'] + 1)
df_encoded['guc_per_hacim'] = df_encoded['Motor Gücü'] / (df_encoded['Motor Hacmi'] + 1)
```

Bu 3 yeni feature **her algoritmanın skorunu 0.01-0.02 arttırır** genelde.

---

## Confusion Matrix — herkes için zorunlu

Raporunda hangi sınıfları karıştırdığını göstermek için:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, best_model.predict(X_test_scaled),
                      labels=['Ekonomik', 'Orta', 'Yuksek', 'Premium'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Ekonomik', 'Orta', 'Yuksek', 'Premium'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix — [Senin Algoritman]')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix_[algoritma].png', dpi=150)
plt.show()
```
