from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib


df = pd.read_csv("EJERCICIO ARBOLES DE DECISION.csv")

cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

for col in cols_with_invalid_zeros:
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

criterios = ['gini', 'entropy', 'log_loss']
for criterio in criterios:
    print(f"\n=== Árbol con criterio: {criterio.upper()} ===")
    clf = DecisionTreeClassifier(criterion=criterio, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

param_grid = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=42),
                    param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Mejores parámetros encontrados:", grid.best_params_)
print("Mejor accuracy en validación cruzada:", grid.best_score_)


mejor_modelo = grid.best_estimator_
y_pred_opt = mejor_modelo.predict(X_test)

print("Accuracy en test:", accuracy_score(y_test, y_pred_opt))
print(classification_report(y_test, y_pred_opt))

cm = confusion_matrix(y_test, y_pred_opt)
ConfusionMatrixDisplay(cm, display_labels=mejor_modelo.classes_).plot(cmap="Blues")
plt.title("Matriz de confusión - Árbol optimizado")
plt.show()

'''
joblib.dump(mejor_modelo, "modelo_arbol_diabetes.pkl")
print("Modelo guardado como modelo_arbol_diabetes.pkl")
'''