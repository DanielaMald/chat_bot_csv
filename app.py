import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chatbot CSV Inteligente", page_icon="🤖", layout="centered")

# CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f8ff 0%, #fff 100%);
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    h1 {
        color: #1a73e8;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot CSV Inteligente")

archivo = st.file_uploader("📂 Sube tu archivo CSV", type="csv")

TOP_N = 5

def es_columna_numerica(serie):
    return pd.api.types.is_numeric_dtype(serie)

def es_columna_fecha(serie):
    return pd.api.types.is_datetime64_any_dtype(serie)

def es_columna_categorica(serie):
    return (serie.dtype == 'object' or pd.api.types.is_string_dtype(serie)) and (serie.nunique() < len(serie)*0.5)

if archivo:
    df = pd.read_csv(archivo)
    st.dataframe(df)

    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore', infer_datetime_format=True)
        except:
            pass

    filas_texto = df.astype(str).agg(' | '.join, axis=1).tolist()

    @st.cache_resource(show_spinner=True)
    def cargar_modelo():
        return SentenceTransformer('all-MiniLM-L6-v2')

    modelo = cargar_modelo()
    embeddings_filas = modelo.encode(filas_texto, convert_to_tensor=True)
    embeddings_columnas = modelo.encode(df.columns.tolist(), convert_to_tensor=True)

    pregunta = st.text_input("💬 Haz una pregunta sobre los datos:")

    if pregunta:
        pregunta_lower = pregunta.lower()
        pregunta_emb = modelo.encode(pregunta, convert_to_tensor=True)
        similitudes_columnas = cosine_similarity(pregunta_emb.reshape(1, -1), embeddings_columnas)[0]
        columnas_ordenadas = df.columns[np.argsort(similitudes_columnas)[::-1]]

        if match := re.search(r'(mayor|menor|m[aá]s|menos|superior|inferior)\s+(?:a|de)?\s*(\d+(\.\d+)?)', pregunta_lower):
            operador, valor, _ = match.groups()
            valor = float(valor)
            for col in columnas_ordenadas:
                if not es_columna_numerica(df[col]):
                    continue
                try:
                    if 'mayor' in operador or 'más' in operador or 'superior' in operador:
                        filtrado = df[df[col] > valor]
                    else:
                        filtrado = df[df[col] < valor]
                    if not filtrado.empty:
                        st.success(f"🔎 Filtrado con '{col}' {operador} {valor}: {len(filtrado)} resultados")
                        st.dataframe(filtrado)
                        break
                except:
                    continue

        elif re.search(r'(cu[aá]l|qu[eé])\s+(es|fue)?\s*(el|la)?\s*[\w\s]\s(m[aá]s|mayor|menos|menor)', pregunta_lower):
            # Checar si se refiere a longitud de texto
            if 'largo' in pregunta_lower or 'extenso' in pregunta_lower or 'longitud' in pregunta_lower:
                for col in columnas_ordenadas:
                    if not pd.api.types.is_string_dtype(df[col]):
                        continue
                    try:
                        idx = df[col].astype(str).str.len().idxmax()
                        fila = df.loc[[idx]]
                        st.success(f"🔤 Resultado con texto más largo en '{col}':")
                        st.dataframe(fila)
                        break
                    except:
                        continue
            else:
                for col in columnas_ordenadas:
                    if not es_columna_numerica(df[col]):
                        continue
                    try:
                        if df[col].isnull().all():
                            continue
                        buscar_max = any(p in pregunta_lower for p in ['más', 'mayor', 'alto', 'caro', 'grande', 'pesado', 'calorías', 'cantidad'])
                        if buscar_max:
                            idx = df[col].idxmax()
                        else:
                            idx = df[col].idxmin()
                        fila = df.loc[[idx]]
                        st.success(f"🔝 Resultado usando la columna numérica '{col}':")
                        st.dataframe(fila)
                        break
                    except:
                        continue

        elif match := re.search(r'(antes|después|despues)\s+de\s+(\d{4}-\d{2}-\d{2})', pregunta_lower):
            operador, fecha_str = match.groups()
            fecha = pd.to_datetime(fecha_str)
            for col in columnas_ordenadas:
                if not es_columna_fecha(df[col]):
                    continue
                try:
                    if 'antes' in operador:
                        filtrado = df[df[col] < fecha]
                    else:
                        filtrado = df[df[col] > fecha]
                    if not filtrado.empty:
                        st.success(f"📅 Resultados con fecha {operador} de {fecha_str} en '{col}':")
                        st.dataframe(filtrado)
                        break
                except:
                    continue

        elif any(pal in pregunta_lower for pal in ['cuántos', 'cuantas', 'cuantos', 'cantidad de', 'número de', 'cuenta', 'listado de', 'qué tipos', 'qué clases', 'categorías', 'tipos', 'clases', 'lista de']):
            for col in columnas_ordenadas:
                if not es_columna_categorica(df[col]):
                    continue
                try:
                    conteo = df[col].value_counts().head(TOP_N)
                    if len(conteo) > 1:
                        st.success(f"📊 Conteo de categorías en '{col}':")
                        st.dataframe(conteo)
                        break
                except:
                    continue

        else:
            encontrado = False
            for col in columnas_ordenadas:
                try:
                    coincidencias = df[df[col].astype(str).str.lower().str.contains(pregunta_lower)]
                    if not coincidencias.empty:
                        st.success(f"🔍 Coincidencias encontradas en '{col}':")
                        st.dataframe(coincidencias)
                        encontrado = True
                        break
                except:
                    continue
            if not encontrado:
                similitudes_filas = cosine_similarity(pregunta_emb.reshape(1, -1), embeddings_filas)[0]
                idx = np.argmax(similitudes_filas)
                if similitudes_filas[idx] > 0.45:
                    st.info("🤖 Resultado más cercano encontrado:")
                    st.dataframe(df.loc[[idx]])
                else:
                    st.warning("❌ No encontré información relacionada con tu pregunta.")
