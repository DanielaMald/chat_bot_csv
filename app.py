import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chatbot-csv", page_icon="🌸", layout="centered")

#CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ffe6f0 0%, #fff0f5 100%);
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #D6CCD8FF;
    }
    h1 {
        color: #d147a3;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .stTextInput>div>div>input {
        border-radius: 15px;
        border: 2px solid #d147a3;
        padding: 10px 15px;
        font-size: 1.1rem;
        color: #D3C6D6FF;
    }
    div.stButton > button {
        background-color: #d147a3;
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
        font-size: 1.1rem;
        width: 100%;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #a2337a;
        cursor: pointer;
    }
    .stDataFrame {
        border-radius: 15px;
        border: 2px solid #d147a3;
        padding: 10px;
        margin-top: 15px;
        background: white;
        color: #DBD1DDFF !important;
    }
    .stAlert {
        background-color: #131112FF !important;
        border-left: 6px solid #d147a3 !important;
        color: #4b2354 !important;
        border-radius: 10px;
        font-size: 1.1rem;
        padding: 15px;
        margin-top: 20px;
        white-space: pre-wrap;
    }
    </style>
""", unsafe_allow_html=True)

# APP
st.title("🌸 Chatbot-csv 🌸")

archivo = st.file_uploader("📁 Sube cualquier CSV", type="csv")

if archivo:
    df = pd.read_csv(archivo)
    st.dataframe(df)

    filas_texto = df.astype(str).agg(' | '.join, axis=1).tolist()

    @st.cache_resource(show_spinner=True)
    def cargar_modelo():
        return SentenceTransformer('all-MiniLM-L6-v2')

    model = cargar_modelo()

    @st.cache_resource(show_spinner=True)
    def crear_embeddings(textos):
        return model.encode(textos, convert_to_tensor=True)

    embeddings_filas = crear_embeddings(filas_texto)

    pregunta = st.text_input("💬 Haz cualquier pregunta sobre los datos:")

    if pregunta:
        pregunta_lower = pregunta.lower()

        # RESPUESTAS DIRECTAS
        if "cuántas filas" in pregunta_lower or "cuantas filas" in pregunta_lower:
            st.info(f"📊 El archivo tiene **{len(df)} filas**.")

        elif "cuántas columnas" in pregunta_lower or "cuantos campos" in pregunta_lower or "cuantas columnas" in pregunta_lower:
            st.info(f"📊 El archivo tiene **{len(df.columns)} columnas**.")

        elif "cómo se llaman las columnas" in pregunta_lower or "nombres de columnas" in pregunta_lower or "campos" in pregunta_lower:
            st.info("📋 Los campos son:\n\n- " + "\n- ".join(df.columns))

        elif "hay una columna" in pregunta_lower or "existe el campo" in pregunta_lower:
            palabras = pregunta_lower.split()
            for palabra in palabras:
                if palabra in [c.lower() for c in df.columns]:
                    st.success(f"✅ Sí, existe una columna llamada '{palabra}'.")
                    break
            else:
                st.warning("❌ No encontré una columna con ese nombre.")

       
        elif ("qué" in pregunta_lower or "cuales" in pregunta_lower) and ("son" in pregunta_lower):
            match = re.search(r"(\w+)\s+son\s+(\w+)", pregunta_lower)
            if match:
                categoria, valor = match.groups()
                col_match = [col for col in df.columns if categoria.lower() in col.lower()]
                if col_match:
                    col = col_match[0]
                    filtro = df[df[col].astype(str).str.lower().str.contains(valor.lower())]
                    st.success(f"🔍 Hay **{len(filtro)}** registros donde '{col}' contiene '{valor}'.")
                    st.dataframe(filtro)
                else:
                    st.warning(f"⚠️ No encontré una columna que coincida con '{categoria}'.")

        
        elif "cuestan más de" in pregunta_lower or "precio mayor a" in pregunta_lower:
            match = re.search(r"(cuestan más de|precio mayor a)\s+(\d+(\.\d+)?)", pregunta_lower)
            if match:
                valor = float(match.group(2))
                col_match = [col for col in df.columns if "precio" in col.lower() or "costo" in col.lower()]
                if col_match:
                    col = col_match[0]
                    filtrado = df[df[col] > valor]
                    st.success(f"💰 Hay **{len(filtrado)}** registros con '{col}' mayor a {valor}.")
                    st.dataframe(filtrado)
                else:
                    st.warning("⚠️ No se encontró una columna de precio o costo.")

      
        elif "stock mayor a" in pregunta_lower:
            match = re.search(r"stock mayor a\s+(\d+)", pregunta_lower)
            if match:
                valor = int(match.group(1))
                col_match = [col for col in df.columns if "stock" in col.lower() or "cantidad" in col.lower()]
                if col_match:
                    col = col_match[0]
                    filtrado = df[df[col] > valor]
                    st.success(f"📦 Hay **{len(filtrado)}** registros con '{col}' mayor a {valor}.")
                    st.dataframe(filtrado)
                else:
                    st.warning("⚠️ No se encontró una columna de stock o cantidad.")

       
        elif "antes de" in pregunta_lower:
            match = re.search(r"antes de\s+(\d{4}-\d{2}-\d{2})", pregunta_lower)
            if match:
                fecha = match.group(1)
                col_match = [col for col in df.columns if "fecha" in col.lower() or "ingreso" in col.lower()]
                if col_match:
                    col = col_match[0]
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    filtrado = df[df[col] < pd.to_datetime(fecha)]
                    st.success(f"📅 Hay **{len(filtrado)}** registros con '{col}' antes de {fecha}.")
                    st.dataframe(filtrado)
                else:
                    st.warning("⚠️ No se encontró columna de fecha.")

      
        elif "estado" in pregunta_lower or "status" in pregunta_lower:
            for valor in ["pendiente", "cancelado", "no pagado", "activo", "inactivo"]:
                if valor in pregunta_lower:
                    col_match = [col for col in df.columns if "estado" in col.lower() or "status" in col.lower()]
                    if col_match:
                        col = col_match[0]
                        filtrado = df[df[col].astype(str).str.lower().str.contains(valor)]
                        st.success(f"📋 Hay **{len(filtrado)}** registros con '{col}' igual a '{valor}'.")
                        st.dataframe(filtrado)
                    else:
                        st.warning("⚠️ No se encontró columna de estado o status.")
                    break

        else:
            emb_pregunta = model.encode(pregunta, convert_to_tensor=True)
            similitudes = cosine_similarity(emb_pregunta.reshape(1, -1), embeddings_filas)[0]
            idx_mejor = np.argmax(similitudes)
            mejor_similitud = similitudes[idx_mejor]

            UMBRAL_SIMILITUD = 0.4
            if mejor_similitud >= UMBRAL_SIMILITUD:
                st.write("✅ Fila más relevante encontrada:")
                st.dataframe(df.iloc[[idx_mejor]])
                st.write(f"🔍 Similitud: {mejor_similitud:.3f}")
            else:
                st.warning("❌ No encontré información relacionada con tu pregunta en el archivo.")
