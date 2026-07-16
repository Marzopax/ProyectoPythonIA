import streamlit as st
import pandas as pd
from transformers import pipeline

@st.cache_resource
def cargar_modelo():
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

classifier = cargar_modelo()
CATEGORIAS = ["Hardware", "Software", "Redes", "Otro"]

st.title("🎫 Sistema Inteligente de Soporte (Procesamiento Local)")
st.write("Subí un archivo CSV con tus requerimientos. El sistema limpiará los datos y asignará el área correspondiente usando IA local.")

UMBRAL_CONFIANZA = st.slider(
    "🎚️ Umbral de confianza mínimo",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="Registros con confianza por debajo de este valor se descartan del informe final."
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
                            CATEGORIAS,
                            hypothesis_template="Este ticket trata sobre un problema de {}."
                        )
                        return res['labels'][0], res['scores'][0]

                    resultados = df[target_col].apply(clasificar_texto)
                    df['Area_Asignada'] = resultados.apply(lambda x: x[0])
                    df['Confianza'] = resultados.apply(lambda x: round(x[1], 3))

                    # Guardamos resultados en session_state para que el slider
                    # pueda refiltrar sin reclasificar
                    st.session_state['df_clasificado'] = df

        if 'df_clasificado' in st.session_state:
            df_clasificado = st.session_state['df_clasificado']
            total_antes = len(df_clasificado)

            df_filtrado = df_clasificado[
                (df_clasificado['Area_Asignada'] != "Otro") &
                (df_clasificado['Confianza'] >= UMBRAL_CONFIANZA)
            ].copy()
            descartados = total_antes - len(df_filtrado)

            st.success(f"¡Análisis completado! {descartados} registro(s) descartado(s) por no relacionarse con soporte técnico.")
            st.dataframe(df_filtrado)

            csv = df_filtrado.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_soporte_ia.csv", "text/csv")

        elif target_col not in df.columns:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
