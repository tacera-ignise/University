import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Zadanie 1
# 1. Generowanie danych
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, n_classes=2, random_state=42)

# 2. Wizualizacja danych
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.title("Przykładowa wizualizacja zbioru losowych obiektów w 2 klasach")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.show()

# 3. Lista klasyfikatorów
classifiers = {
    "GaussianNB": GaussianNB(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# 4-5. Podział danych na treningowe i testowe, uczenie i testowanie klasyfikatorów
results = []

for name, clf in classifiers.items():
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    auc_scores = []
    
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
        
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
        recall_scores.append(metrics.recall_score(y_test, y_pred))
        precision_scores.append(metrics.precision_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
        auc_scores.append(metrics.roc_auc_score(y_test, y_proba))
        
    results.append({
        "Classifier": name,
        "Accuracy": np.mean(accuracy_scores),
        "Recall": np.mean(recall_scores),
        "Precision": np.mean(precision_scores),
        "F1": np.mean(f1_scores),
        "AUC": np.mean(auc_scores)
    })

results_df = pd.DataFrame(results)
print(results_df)

# 6. Wizualizacja błędów klasyfikacji dla ostatniej iteracji
name, clf = list(classifiers.items())[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c=(y_test == y_pred), cmap='bwr', alpha=0.7)
plt.title(f"Błędy klasyfikacji dla {name}")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.show()

# 7. Krzywa ROC dla ostatniej iteracji
y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Krzywa ROC dla {name}")
plt.legend(loc="lower right")
plt.show()

# 8. Krzywa dyskryminacyjna
xx, yy = np.meshgrid(np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.01),
                     np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap='bwr')
plt.title(f"Krzywa dyskryminacyjna dla {name}")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.show()

# Zadanie 2
# 1. Generowanie danych
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, n_classes=2, random_state=42)

# 2. Wybór klasyfikatora
classifier_name = "KNeighborsClassifier"
classifier = KNeighborsClassifier()

# 3. Przestrzeń parametrów
param_grid = {
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "p": [1, 2]
    }
}

# 4. Optymalizacja parametrów
grid_search = GridSearchCV(classifier, param_grid[classifier_name], scoring='roc_auc', cv=5)
grid_search.fit(X, y)

# 5. Wizualizacja wpływu parametrów
results = pd.DataFrame(grid_search.cv_results_)
pivot_table = results.pivot_table(values="mean_test_score", index="param_n_neighbors", columns="param_p")

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(pivot_table, cmap='YlGnBu')
fig.colorbar(cax)

ax.set_xticks(np.arange(len(pivot_table.columns)))
ax.set_yticks(np.arange(len(pivot_table.index)))

ax.set_xticklabels(pivot_table.columns)
ax.set_yticklabels(pivot_table.index)

for (i, j), val in np.ndenumerate(pivot_table.values):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center')

plt.title("Wykres wpływu parametrów n_neighbors i p na AUC")
plt.xlabel("p")
plt.ylabel("n_neighbors")
plt.show()

# 6-7. Testowanie klasyfikatora z optymalnymi parametrami
best_params = grid_search.best_params_
classifier.set_params(**best_params)

accuracy_scores = []
recall_scores = []
precision_scores = []
f1_scores = []
auc_scores = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]
    
    accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
    recall_scores.append(metrics.recall_score(y_test, y_pred))
    precision_scores.append(metrics.precision_score(y_test, y_pred))
    f1_scores.append(metrics.f1_score(y_test, y_pred))
    auc_scores.append(metrics.roc_auc_score(y_test, y_proba))

results = {
    "Accuracy": np.mean(accuracy_scores),
    "Recall": np.mean(recall_scores),
    "Precision": np.mean(precision_scores),
    "F1": np.mean(f1_scores),
    "AUC": np.mean(auc_scores)
}

results_df = pd.DataFrame([results])
print(results_df)

# 8. Krzywa ROC i krzywa dyskryminacyjna
y_proba = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Krzywa ROC dla {classifier_name}")
plt.legend(loc="lower right")
plt.show()

# Krzywa dyskryminacyjna
xx, yy = np.meshgrid(np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.01),
                     np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.01))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap='bwr')
plt.title(f"Krzywa dyskryminacyjna dla {classifier_name}")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.show()
