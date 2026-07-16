import streamlit as st
import pandas as pd
from transformers import pipeline

# ============================================================
# CARGA DEL MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    # @st.cache_resource evita recargar el modelo en cada interacción
    # (slider, botón, etc.). Se carga una sola vez por sesión de servidor.
    # mDeBERTa-v3-base-mnli-xnli: modelo NLI multilingüe (soporta español),
    # usado para zero-shot classification sin necesidad de fine-tuning.
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

classifier = cargar_modelo()

# ============================================================
# CATEGORÍAS DE CLASIFICACIÓN
# ============================================================

# Cada categoría incluye una descripción extendida (no solo la palabra sola)
# para darle más contexto semántico al modelo y mejorar la precisión,
# especialmente en español.
CATEGORIAS_DESC = {
    "Hardware": "Falla física de un equipo o dispositivo: impresora, computadora, monitor, teclado, mouse, cable, "
                "escáner o periférico que no enciende, no responde, hace ruido, se traba o está roto físicamente",
    "Software": "Falla de un programa o aplicación instalada: error al abrir, se cierra solo, licencia vencida, "
                "actualización fallida, mensaje de error en pantalla, problema al instalar un programa",
    "Redes": "Falla de conectividad: sin acceso a internet, wifi que no conecta, VPN caída, conexión lenta o "
             "intermitente, el router o módem sin luces o sin señal",
    "Cuentas y Accesos": "Problemas de inicio de sesión, usuario o contraseña: no puede loguearse, olvidó la "
                          "contraseña, cuenta bloqueada, permisos de acceso denegados, correo que no llega",
}
CATEGORIAS_LABELS = list(CATEGORIAS_DESC.keys())       # Nombres cortos (para mostrar en el resultado)
CATEGORIAS_HIPOTESIS = list(CATEGORIAS_DESC.values())  # Descripciones largas (para alimentar al modelo)

# ============================================================
# INTERFAZ: TÍTULO Y DESCRIPCIÓN
# ============================================================

st.title("Sistema Inteligente de Soporte")
st.write("Subi un CSV, el sistema lo limpiara y organizara")
st.write("Con los sliders podes controlar los umbrales y longitud para filtrar descripcion")

# ============================================================
# SLIDERS DE CALIBRACIÓN
# ============================================================
# Tres controles independientes para ajustar el filtrado sin
# necesidad de tocar código ni reclasificar con el modelo.

col1, col2, col3 = st.columns(3)

with col1:
    # Umbral de confianza para descripciones "normales" (largas)
    UMBRAL_CONFIANZA = st.slider(
        "Umbral confianza (normales)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Aplica a descripciones iguales o más largas que el umbral de longitud."
    )
with col2:
    # Umbral de confianza para descripciones "cortas", normalmente
    # más bajo porque el modelo tiene menos texto para decidir
    UMBRAL_CORTO = st.slider(
        "Umbral confianza (cortas)",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Aplica a descripciones más cortas que el umbral de longitud."
    )
with col3:
    # Cantidad de caracteres que define el punto de corte entre
    # "descripción corta" y "descripción normal"
    LONGITUD_CORTA = st.slider(
        "Umbral de longitud (caracteres)",
        min_value=0, max_value=50, value=10, step=1,
        help="Descripciones con menos caracteres que este valor se consideran 'cortas' y usan el umbral de confianza correspondiente."
    )

# ============================================================
# CARGA DEL ARCHIVO CSV
# ============================================================

def reparar_mojibake(texto):
    # Corrige texto UTF-8 que fue mal decodificado como Latin-1 en el origen
    # del archivo (patrón típico: 'secretarÃ­a' en vez de 'secretaría',
    # 'extraÃ±o' en vez de 'extraño'). Esto ocurre cuando el CSV ya viene
    # corrupto de otro sistema, antes de llegar a esta app.
    # Si el texto ya está bien codificado, encode('latin1') produce bytes
    # que no forman UTF-8 válido y decode('utf-8') falla -> se deja igual.
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto

uploaded_file = st.file_uploader("Subir archivo de requerimientos (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Intenta leer en UTF-8 primero; si falla, reintenta en Latin-1
        # (cubre CSVs exportados desde Excel en Windows en español)
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin1')

        target_col = 'descripcion'
        if target_col in df.columns:

            # -------------------- LIMPIEZA CON PANDAS --------------------
            df[target_col] = df[target_col].astype(str).apply(reparar_mojibake)  # repara mojibake heredado del origen
            df.dropna(subset=[target_col], inplace=True)          # elimina filas sin descripción
            df.drop_duplicates(subset=[target_col], inplace=True) # elimina descripciones duplicadas
            df[target_col] = df[target_col].astype(str).str.strip()  # recorta espacios sobrantes
            df = df[df[target_col].str.len() > 3]                 # descarta textos vacíos/basura (ej. "a", "-")

            st.write("### Datos normalizados con Pandas:", df.head())

            # -------------------- CLASIFICACIÓN CON IA --------------------
            if st.button("🚀 Clasificar requerimientos con IA"):
                with st.spinner("Clasificando localmente..."):

                    def clasificar_texto(texto):
                        # multi_label=True: cada categoría se evalúa de forma
                        # independiente (no compiten por sumar 1 entre todas),
                        # lo que da scores más representativos por categoría.
                        res = classifier(
                            texto,
                            CATEGORIAS_HIPOTESIS,
                            multi_label=True,
                            # Template en español y orientado al dominio: mejora la
                            # comparación semántica frente al template genérico en inglés
                            hypothesis_template="Este reclamo de soporte técnico trata sobre: {}."
                        )
                        # res['labels'] viene ordenado por score descendente
                        mejor_hipotesis = res['labels'][0]
                        mejor_score = res['scores'][0]
                        segunda_hipotesis = res['labels'][1]
                        segundo_score = res['scores'][1]
                        # Traduce la hipótesis larga de vuelta al nombre corto de categoría
                        idx = CATEGORIAS_HIPOTESIS.index(mejor_hipotesis)
                        categoria = CATEGORIAS_LABELS[idx]
                        idx2 = CATEGORIAS_HIPOTESIS.index(segunda_hipotesis)
                        categoria_alternativa = CATEGORIAS_LABELS[idx2]
                        # gap chico entre el 1er y 2do score = ambigüedad entre dos categorías
                        gap = round(mejor_score - segundo_score, 3)
                        return categoria, mejor_score, categoria_alternativa, gap

                    # Aplica la clasificación fila por fila sobre la columna de descripción
                    resultados = df[target_col].apply(clasificar_texto)
                    df['Area_Asignada'] = resultados.apply(lambda x: x[0])       # categoría asignada
                    df['Confianza'] = resultados.apply(lambda x: round(x[1], 3)) # score del modelo
                    df['Area_Alternativa'] = resultados.apply(lambda x: x[2])    # segunda categoría más probable
                    df['Ambiguedad'] = resultados.apply(lambda x: x[3])          # diferencia entre 1er y 2do score (chico = dudoso)
                    df['Longitud'] = df[target_col].str.len()                    # longitud del texto, usada en el filtro

                    # Se guarda en session_state para poder refiltrar con los
                    # sliders sin tener que volver a correr el modelo
                    st.session_state['df_clasificado'] = df

        # -------------------- FILTRADO Y RESULTADOS --------------------
        if 'df_clasificado' in st.session_state:
            df_clasificado = st.session_state['df_clasificado']
            total_antes = len(df_clasificado)

            # Divide el dataframe en "cortas" y "normales" según LONGITUD_CORTA,
            # y aplica el umbral de confianza correspondiente a cada grupo
            es_corta = df_clasificado['Longitud'] < LONGITUD_CORTA
            pasa_corta = es_corta & (df_clasificado['Confianza'] >= UMBRAL_CORTO)
            pasa_normal = ~es_corta & (df_clasificado['Confianza'] >= UMBRAL_CONFIANZA)

            df_filtrado = df_clasificado[pasa_corta | pasa_normal].copy()
            descartados = total_antes - len(df_filtrado)

            st.success(f"¡Análisis completado! {descartados} registro(s) descartado(s) por no relacionarse con soporte técnico.")
            st.dataframe(df_filtrado)

            # utf-8-sig agrega el BOM (Byte Order Mark) necesario para que
            # Excel en Windows reconozca UTF-8 y no rompa tildes/ñ al abrir el CSV
            csv = df_filtrado.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Descargar CSV Clasificado", csv, "tickets_soporte_ia.csv", "text/csv")

        elif target_col not in df.columns:
            st.error(f"El archivo debe contener una columna llamada '{target_col}'.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
