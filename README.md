# Predicción de Calidad del Vino 🍷

Este proyecto es una práctica completa de **Data Science y Machine Learning** enfocada en predecir la calidad del vino tinto (en una escala de puntuación de expertos) basándose en sus características físico-químicas.

El flujo de trabajo sigue las mejores prácticas de exploración, limpieza y modelado de datos, inspirado en otros pipelines clásicos de análisis.

---

## 📋 Descripción del Dataset

El proyecto utiliza el **Wine Quality Dataset** del repositorio de *UCI Machine Learning*. Este dataset contiene el resultado de pruebas fisicoquímicas a muestras de la variante roja del vino portugués "Vinho Verde".

### Variables (Features)
- **`fixed acidity`**: Ácidos que no se evaporan fácilmente.
- **`volatile acidity`**: Cantidad de ácido acético en el vino.
- **`citric acid`**: Añade frescura y sabor al vino.
- **`residual sugar`**: Azúcar remanente tras finalizar la fermentación.
- **`chlorides`**: Cantidad de sal en el vino.
- **`free sulfur dioxide` / `total sulfur dioxide`**: Dióxido de azufre libre y total (antimicrobiano y antioxidante).
- **`density`**: Densidad relativa a la concentración de alcohol y azúcar.
- **`pH`**: Nivel de acidez (0-14).
- **`sulphates`**: Aditivo relacionado con el SO2.
- **`alcohol`**: Grado alcohólico del vino.

### Etiqueta (Target)
- **`quality`**: Puntuación otorgada por expertos catadores (escala ordinal de 0 a 10).

---

## 📂 Estructura del Proyecto

```text
📁 Wine Quality/
│
├── 📁 data/                        # Contiene los datasets (se generan automáticamente al correr el notebook)
│   ├── wine_original.csv           # Dataset bruto descargado
│   └── vinos_limpios.csv           # Dataset limpio sin duplicados/nulos
│
├── 📁 modelos/                     # Modelos de ML serializados
│   └── modelo_vinos.pkl            # Modelo final (Random Forest) guardado listo para predicciones
│
├── wine_quality_notebook.ipynb     # Notebook principal con EDA y entrenamiento de modelos
└── README.md                       # Documentación del proyecto
```

---

## 🛠️ Tecnologías y Librerías

El análisis está construido enteramente en Python usando:
- **Pandas**: Para la manipulación y limpieza de los datos.
- **Matplotlib & Seaborn**: Para la visualización de la matriz de correlación.
- **Plotly Express**: Para gráficos interactivos de análisis exploratorio (EDA).
- **Scikit-Learn**: Para el preprocesamiento de variables (`StandardScaler`), entrenamiento y evaluación de modelos.

---

## 🤖 Modelos Evaluados

En este notebook se entrena y compara el rendimiento (Accuracy) de los siguientes algoritmos de clasificación multiclase:
1. **K-Nearest Neighbors (KNN)** (Con datos en bruto).
2. **K-Nearest Neighbors (KNN) Normalizado** (Con variables estandarizadas).
3. **Regresión Logística**.
4. **Árboles de Decisión (Decision Tree)** (Con profundidad controlada para visualización).
5. **Random Forest Classifier** (Mejor algoritmo guardado en disco).

---

## 🚀 Cómo Empezar y Ejecutar el Proyecto

Es muy recomendable correr este proyecto en un **entorno virtual** para no afectar los paquetes de tu sistema. 

### 1. Clonar el proyecto e ir a la carpeta
```bash
git clone <URL_DEL_REPOSITORIO>
cd "Wine Quality"
```

### 2. Crear y activar el Entorno Virtual
En Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
En Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar las Dependencias
```bash
pip install pandas plotly matplotlib seaborn scikit-learn notebook nbformat joblib
```

### 4. Lanzar Jupyter Notebook
```bash
jupyter notebook
```
Una vez en el navegador, abre el archivo `wine_quality_notebook.ipynb` y ejecuta todas las celdas paso a paso. Es **obligatorio** correr este notebook primero para que se genere el archivo `modelo_vinos.pkl` en la carpeta `modelos/`.

### 5. Lanzar la App Web (Streamlit)
Una vez generado el modelo, puedes lanzar la interfaz visual interactiva:
```bash
streamlit run app.py
```
Esto abrirá una pestaña en tu navegador web donde podrás jugar con los químicos y ver la predicción en tiempo real.

---

## ✨ Próximos Pasos Posibles
- Probar convertir el problema en una clasificación **binaria** (Vino "Bueno" si `quality` >= 7, "Malo" si es < 7).
- Construir una pequeña aplicación con **Streamlit** que cargue el archivo `modelo_vinos.pkl` para predecir la calidad de un nuevo vino dado sus químicos.
