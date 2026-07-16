import streamlit as st
import pandas as pd
from transformers import pipeline

@st.cache_resource
def cargar_modelo():
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

classifier = cargar_modelo()

CATEGORIAS_DESC = {
    "Hardware": "Hardware: problemas con impresoras, computadoras, monitores, teclados, mouses, cables, dispositivos físicos que no encienden o no funcionan",
    "Software": "Software: problemas con programas, licencias, instalaciones, actualizaciones, aplicaciones que no abren o dan error",
    "Redes": "Redes: problemas de conexión a internet, wifi, módem, router, VPN, sin señal o conexión lenta",
}
CATEGORIAS_LABELS = list(CATEGORIAS_DESC.keys())
CATEGORIAS_HIPOTESIS = list(CATEGORIAS_DESC.values())

st.title("🎫 Sistema Inteligente de Soporte (Procesamiento Local)")
st.write("Subí un archivo CSV con tus requerimientos. El sistema limpiará los datos y asignará el área correspondiente usando IA local.")

col1, col2, col3 = st.columns(3)
with col1:
    UMBRAL_CONFIANZA = st.slider(
        "🎚️ Umbral confianza (normales)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Aplica a descripciones iguales o más largas que el umbral de longitud."
    )
with col2:
    UMBRAL_CORTO = st.slider(
        "🎚️ Umbral confianza (cortas)",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Aplica a descripciones más cortas que el umbral de longitud."
    )
with col3:
    LONGITUD_CORTA = st.slider(
        "📏 Umbral de longitud (caracteres)",
        min_value=0, max_value=50, value=10, step=1,
        help="Descripciones con menos caracteres que este valor se consideran 'cortas' y usan el umbral de confianza correspondiente."
    )

uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin1')

        target_col = 'descripcion'
        if target_col in df.columns:
            df.dropna(subset=[target_col], inplace=True)
            df.drop_duplicates(subset=[target_col], inplace=True)
            df[target_col] = df[target_col].astype(str).str.strip()
            df = df[df[target_col].str.len() > 3]

            st.write("### Datos normalizados con Pandas:", df.head())

            if st.button("🚀 Clasificar requerimientos con IA"):
                with st.spinner("Clasificando localmente..."):

                    def clasificar_texto(texto):
                        res = classifier(
                            texto,
                            CATEGORIAS_HIPOTESIS,
                            multi_label=True
                        )
                        mejor_hipotesis = res['labels'][0]
                        mejor_score = res['scores'][0]
                        idx = CATEGORIAS_HIPOTESIS.index(mejor_hipotesis)
                        categoria = CATEGORIAS_LABELS[idx]
                        return categoria, mejor_score

                    resultados = df[target_col].apply(clasificar_texto)
                    df['Area_Asignada'] = resultados.apply(lambda x: x[0])
                    df['Confianza'] = resultados.apply(lambda x: round(x[1], 3))
                    df['Longitud'] = df[target_col].str.len()

                    st.session_state['df_clasificado'] = df

        if 'df_clasificado' in st.session_state:
            df_clasificado = st.session_state['df_clasificado']
            total_antes = len(df_clasificado)

            es_corta = df_clasificado['Longitud'] < LONGITUD_CORTA
            pasa_corta = es_corta & (df_clasificado['Confianza'] >= UMBRAL_CORTO)
            pasa_normal = ~es_corta & (df_clasificado['Confianza'] >= UMBRAL_CONFIANZA)

            df_filtrado = df_clasificado[pasa_corta | pasa_normal].copy()
            descartados = total_antes - len(df_filtrado)

            st.success(f"¡Análisis completado! {descartados} registro(s) descartado(s) por no relacionarse con soporte técnico.")
            st.dataframe(df_filtrado)

            csv = df_filtrado.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_soporte_ia.csv", "text/csv")

        elif target_col not in df.columns:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
