import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Installs
cells.append(nbf.v4.new_code_cell('%pip install pandas plotly matplotlib seaborn scikit-learn nbformat joblib'))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell('''import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns'''))

# Cell 3: Data loading
cells.append(nbf.v4.new_code_cell('''# El Wine Quality dataset está en el repositorio de UCI Machine Learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# OJO: Los datos vienen separados por punto y coma (;)
df = pd.read_csv(url, sep=';')
print(df.shape)
df.head()'''))

# Cell 4: Save original
cells.append(nbf.v4.new_code_cell('''import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/wine_original.csv", index=False)'''))

# Cell 5: NaNs
cells.append(nbf.v4.new_code_cell('df.isna().sum()'))

# Cell 6: Drop NaNs
cells.append(nbf.v4.new_code_cell('''df = df.dropna()
print(df.shape)'''))

# Cell 7: Duplicates
cells.append(nbf.v4.new_code_cell('''print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)'''))

# Cell 8: Stats
cells.append(nbf.v4.new_code_cell('df[["alcohol", "pH", "volatile acidity", "residual sugar"]].describe()'))

# Cell 9: Target distribution
cells.append(nbf.v4.new_code_cell('df["quality"].value_counts()'))

# Cell 10: Convert target to string for categorical plots
cells.append(nbf.v4.new_code_cell('''# Convertimos 'quality' a texto para que Plotly lo trate como categorías (grupos)
df["quality"] = df["quality"].astype(str)'''))

# Cell 11: Save clean data
cells.append(nbf.v4.new_code_cell('df.to_csv("data/vinos_limpios.csv", index=False)'))

# Cell 12: EDA Boxplot
cells.append(nbf.v4.new_code_cell('''colores = {
    "3": "#4CC9F0",
    "4": "#F2C14E",
    "5": "#F25C54",
    "6": "#B388EB",
    "7": "#1ED760",
    "8": "#FF8C00"
}

# ¿Cómo se relaciona el alcohol con la calidad?
fig = px.box(df, x="quality", y="alcohol", color="quality", color_discrete_map=colores, category_orders={"quality": ["3", "4", "5", "6", "7", "8"]})
fig.show()'''))

# Cell 13: EDA Scatter
cells.append(nbf.v4.new_code_cell('''# ¿Cómo se relaciona la acidez volátil con el alcohol?
fig = px.scatter(df, x="alcohol", y="volatile acidity", color="quality",
                 color_discrete_map=colores, opacity=0.6, category_orders={"quality": ["3", "4", "5", "6", "7", "8"]})
fig.show()'''))

# Cell 14: Correlation matrix
cells.append(nbf.v4.new_code_cell('''plt.figure(figsize=(12, 8))
# Volvemos a hacer numérico 'quality' solo para esta matriz
df_num = df.copy()
df_num["quality"] = df_num["quality"].astype(int)
sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()'''))

# Cell 15: Preparation for models
cells.append(nbf.v4.new_code_cell('''# Definimos las variables predictoras (X) y la variable a predecir (y)
X = df.drop("quality", axis=1)
y = df["quality"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)'''))

# Cell 16: KNN
cells.append(nbf.v4.new_code_cell('''from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)
print(f"Accuracy of KNN: ", acc_knn)'''))

# Cell 17: Standard Scaler & KNN Scaled
cells.append(nbf.v4.new_code_cell('''from sklearn.preprocessing import StandardScaler

escalador = StandardScaler()
escalador.fit(X_train)

X_train_esc = pd.DataFrame(escalador.transform(X_train), columns=X.columns)
X_test_esc = pd.DataFrame(escalador.transform(X_test), columns=X.columns)

knn_esc = KNeighborsClassifier(n_neighbors=5)
knn_esc.fit(X_train_esc, y_train)

pred_knn_esc = knn_esc.predict(X_test_esc)
acc_knn_esc = accuracy_score(y_test, pred_knn_esc)
print(f"Accuracy of KNN with scaled data: ", acc_knn_esc)'''))

# Cell 18: Logistic Regression
cells.append(nbf.v4.new_code_cell('''from sklearn.linear_model import LogisticRegression

logistica = LogisticRegression(max_iter=2000)
logistica.fit(X_train_esc, y_train)

pred_log = logistica.predict(X_test_esc)
acc_log = accuracy_score(y_test, pred_log)
print("Accuracy regresión logística:", acc_log)'''))

# Cell 19: Decision Tree
cells.append(nbf.v4.new_code_cell('''from sklearn.tree import DecisionTreeClassifier, plot_tree

arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol.fit(X_train, y_train)

pred_arbol = arbol.predict(X_test)
acc_arbol = accuracy_score(y_test, pred_arbol)
print("Accuracy árbol:", acc_arbol)'''))

# Cell 20: Random Forest
cells.append(nbf.v4.new_code_cell('''from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)
print("Accuracy Random Forest:", acc_rf)'''))

# Cell 21: Model Comparison
cells.append(nbf.v4.new_code_cell('''resultados = pd.DataFrame({
    "modelo": ["KNN", "KNN normalizado", "Regresión logística", "Árbol (4 niveles)", "Random Forest"],
    "accuracy": [acc_knn, acc_knn_esc, acc_log, acc_arbol, acc_rf],
})

fig = px.bar(resultados, x="modelo", y="accuracy", color="modelo", text_auto=".2f")
fig.show()'''))

# Cell 22: Feature Importances
cells.append(nbf.v4.new_code_cell('''importancias = pd.DataFrame({
    "columna": X.columns,
    "importancia": rf.feature_importances_,
})

importancias = importancias.sort_values("importancia")

fig = px.bar(importancias, x="importancia", y="columna", orientation="h", title="Importancia de Variables (Random Forest)")
fig.show()'''))

# Cell 23: Save Model
cells.append(nbf.v4.new_code_cell('''import joblib
import os

os.makedirs("modelos", exist_ok=True)
joblib.dump(rf, "modelos/modelo_vinos.pkl")'''))

# Cell 24: Test Model
cells.append(nbf.v4.new_code_cell('''modelo = joblib.load("modelos/modelo_vinos.pkl")

# Tomamos un vino de ejemplo del conjunto de test
vino_ejemplo = X_test.iloc[[0]]
display(vino_ejemplo)

prediccion = modelo.predict(vino_ejemplo)
print("Calidad predicha:", prediccion[0])
print("Calidad real:", y_test.iloc[0])'''))


nb['cells'] = cells

with open('wine_quality_notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
