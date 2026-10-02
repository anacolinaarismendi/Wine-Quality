import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Calidad del Vino",
    page_icon="🍷",
    layout="wide"
)

# --- PALETA DE COLORES VINO BURDEOS ---
# Tonos de más claro (peor calidad) a más oscuro/profundo (mejor calidad)
colores_vino = {
    "3": "#F8E5E5", # Blanco/Rosado muy pálido
    "4": "#D88C9A", # Rosado
    "5": "#A93F55", # Rojo cereza
    "6": "#8C1C13", # Rojo rubí
    "7": "#721121", # Burdeos clásico
    "8": "#4E070C"  # Granate profundo
}

escala_continua_vino = ["#F8E5E5", "#D88C9A", "#A93F55", "#8C1C13", "#721121", "#4E070C"]

# Carga de datos base (para comparación visual y correlaciones)
@st.cache_data
def cargar_datos_base():
    try:
        df = pd.read_csv("data/vinos_limpios.csv")
        df["quality_str"] = df["quality"].astype(str) # Columna auxiliar para gráficos discretos
        
        # Calculamos los promedios de los vinos excelentes (calidad >= 7)
        buenos = df[df["quality"] >= 7].drop(["quality", "quality_str"], axis=1, errors='ignore').mean()
        return df, buenos
    except:
        return None, None

# Carga del modelo
@st.cache_resource
def load_model():
    return joblib.load("modelos/modelo_vinos.pkl")

try:
    modelo = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo. ¿Ejecutaste el notebook primero? Error: {e}")
    st.stop()

df_completo, promedios_buenos = cargar_datos_base()

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.title("🧪 Ajusta los Químicos")
st.sidebar.write("Desliza para formular tu vino ideal.")

st.sidebar.markdown("### 🍋 Acidez")
fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.3, 0.1, help="Ácidos fijos. Cantidades muy altas hacen el vino demasiado duro.")
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 2.0, 0.52, 0.01, help="Ácido acético. Un exceso hace que el vino sepa a vinagre.")
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.27, 0.01, help="Aporta frescura y sabor, presente en cantidades pequeñas.")
pH = st.sidebar.slider("pH", 2.7, 4.0, 3.3, 0.01, help="La mayoría de los vinos tienen entre 3.0 y 4.0 de pH. Mide la acidez.")

st.sidebar.markdown("### 🧊 Aditivos y Compuestos")
residual_sugar = st.sidebar.slider("Residual Sugar", 0.9, 15.0, 2.5, 0.1, help="Azúcar restante post fermentación.")
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.6, 0.08, 0.001, format="%.3f", help="Básicamente la cantidad de sal en el vino.")
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.65, 0.01, help="Aditivo común que combate bacterias y oxidación.")

st.sidebar.markdown("### 💨 Dióxido de Azufre (SO2)")
free_sulfur_dioxide = st.sidebar.slider("Free SO2", 1, 72, 15, 1, help="Previene crecimiento de microbios y oxidación.")
total_sulfur_dioxide = st.sidebar.slider("Total SO2", 6, 289, 46, 1, help="Dióxido de azufre total.")

st.sidebar.markdown("### 🌡️ Densidad y Alcohol")
density = st.sidebar.number_input("Density", 0.990, 1.004, 0.996, 0.001, format="%.4f", help="Depende fuertemente del alcohol y azúcar.")
alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.4, 0.1, help="Porcentaje de alcohol volumétrico.")

boton_predecir = st.sidebar.button("🔮 Predecir Calidad", use_container_width=True, type="primary")


# --- PANTALLA PRINCIPAL ---
st.title("🍷 Análisis y Predicción de Vinos")

# Usamos pestañas para organizar mejor el contenido
tab1, tab2 = st.tabs(["🔮 Predicción Interactiva", "📊 Panel de Correlaciones"])

# --- PESTAÑA 1: PREDICCIÓN ---
with tab1:
    st.write("Ajusta los valores en el panel lateral izquierdo y descubre la calificación que nuestro modelo de **Random Forest** le otorgaría a tu vino.")

    if boton_predecir:
        # Preparar DataFrame
        datos_vino = pd.DataFrame({
            "fixed acidity": [fixed_acidity],
            "volatile acidity": [volatile_acidity],
            "citric acid": [citric_acid],
            "residual sugar": [residual_sugar],
            "chlorides": [chlorides],
            "free sulfur dioxide": [free_sulfur_dioxide],
            "total sulfur dioxide": [total_sulfur_dioxide],
            "density": [density],
            "pH": [pH],
            "sulphates": [sulphates],
            "alcohol": [alcohol]
        })
        
        # Predicción
        prediccion = int(modelo.predict(datos_vino)[0])
        
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.subheader("Tu Resultado")
            if prediccion >= 7:
                estado = "Excelente 🏆"
                delta = "Alta Gama"
                st.balloons()
            elif prediccion in [5, 6]:
                estado = "Promedio 👍"
                delta = "Calidad Estándar"
                delta_color = "off"
            else:
                estado = "Baja Calidad 📉"
                delta = "Podría mejorar"
                delta_color = "inverse"
                
            st.metric(label="Calidad Predicha (Escala 3 al 8)", value=f"{prediccion} / 8", delta=delta, delta_color=delta_color)
            st.markdown(f"**Veredicto:** {estado}")
            
        with col_res2:
            if promedios_buenos is not None:
                features = ["alcohol", "fixed acidity", "pH", "sulphates", "volatile acidity", "citric acid"]
                valores_usuario = datos_vino[features].iloc[0].values.tolist()
                valores_buenos = promedios_buenos[features].values.tolist()
                
                fig = go.Figure()
                
                # Radar del vino promedio excelente (color dorado para contrastar o un burdeos clarito)
                fig.add_trace(go.Scatterpolar(
                      r=valores_buenos + [valores_buenos[0]],
                      theta=features + [features[0]],
                      fill='toself',
                      name='Vinos Excelentes (Media)',
                      line_color='#D88C9A',
                      fillcolor='rgba(216, 140, 154, 0.4)' 
                ))
                
                # Radar del vino del usuario (Burdeos profundo)
                fig.add_trace(go.Scatterpolar(
                      r=valores_usuario + [valores_usuario[0]],
                      theta=features + [features[0]],
                      fill='toself',
                      name='Tu Vino Creado',
                      line_color='#4E070C',
                      fillcolor='rgba(78, 7, 12, 0.6)'
                ))
                
                fig.update_layout(
                  polar=dict(
                    radialaxis=dict(visible=True, range=[0, 15])
                  ),
                  showlegend=True,
                  title="Comparativa: Tu Vino vs Vinos Excelentes",
                  margin=dict(t=40, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("**Sobre este gráfico (Radar):** Compara tu receta contra el promedio de todos los vinos con nota 7 u 8. Observa cómo el 'alcohol' suele ser un factor determinante.", icon="🌿")
            else:
                st.warning("No se encontró el dataset base para dibujar la gráfica comparativa.")
    else:
        st.success("👈 Pulsa **Predecir Calidad** en el menú izquierdo para comenzar.", icon="🌿")

# --- PESTAÑA 2: CORRELACIONES ---
with tab2:
    st.write("Explora cómo se relacionan los diferentes componentes químicos del vino entre sí.")
    
    if df_completo is not None:
        # Matriz de Correlación
        correlacion = df_completo.drop("quality_str", axis=1).corr()
        
        # Heatmap usando la escala de tonos burdeos
        fig_corr = px.imshow(correlacion, 
                             text_auto=".2f", 
                             aspect="auto", 
                             color_continuous_scale=escala_continua_vino,
                             title="Mapa de Calor (Heatmap) de Correlaciones")
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.success("**Sobre este gráfico:** Los tonos burdeos más oscuros indican una **correlación positiva fuerte** (si una sube, la otra también, como el alcohol y la calidad). Los tonos más claros indican **correlación negativa** (si una sube, la otra baja, como la acidez volátil y la calidad).", icon="🌿")
        
        st.divider()
        
        # Gráfico adicional de contexto
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_box = px.box(df_completo, x="quality_str", y="alcohol", color="quality_str", 
                             title="Distribución de Alcohol por Calidad",
                             color_discrete_map=colores_vino,
                             category_orders={"quality_str": ["3", "4", "5", "6", "7", "8"]})
            fig_box.update_layout(showlegend=False, xaxis_title="Calidad", yaxis_title="Alcohol")
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("📈 **Cajas y Bigotes:** A medida que oscurece el burdeos (mejor calidad), sube la media de alcohol.")
            
        with col_c2:
            fig_scatter = px.scatter(df_completo, x="volatile acidity", y="alcohol", color="quality_str",
                                     title="Alcohol vs Acidez Volátil",
                                     color_discrete_map=colores_vino,
                                     category_orders={"quality_str": ["3", "4", "5", "6", "7", "8"]})
            fig_scatter.update_layout(xaxis_title="Acidez Volátil", yaxis_title="Alcohol", legend_title="Calidad")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("📉 **Dispersión:** Los vinos Burdeos más oscuros (mejores puntuados) se agrupan en la zona de alto alcohol y baja acidez volátil.")

    else:
        st.error("Archivo 'vinos_limpios.csv' no encontrado en 'data/'. Por favor ejecuta el notebook primero.")
