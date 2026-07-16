import streamlit as st
import pandas as pd
from transformers import pipeline

@st.cache_resource
def cargar_modelo():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = cargar_modelo()
CATEGORIAS = ["Hardware", "Software", "Redes", "Otro"]
UMBRAL_CONFIANZA = 0.55  # ajustable

st.title("🎫 Sistema Inteligente de Soporte (Procesamiento Local)")
st.write("Subí un archivo CSV con tus requerimientos. El sistema limpiará los datos y asignará el área correspondiente usando IA local.")

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
            df = df[df[target_col].str.len() > 3]  # descarta textos vacíos/basura

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

                    total_antes = len(df)
                    df_filtrado = df[
                        (df['Area_Asignada'] != "Otro") &
                        (df['Confianza'] >= UMBRAL_CONFIANZA)
                    ].copy()
                    descartados = total_antes - len(df_filtrado)

                    st.success(f"¡Análisis completado! {descartados} registro(s) descartado(s) por no relacionarse con soporte técnico.")
                    st.dataframe(df_filtrado)

                    csv = df_filtrado.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_soporte_ia.csv", "text/csv")
        else:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
