import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------- DISEÑO ROSITA ---------
st.set_page_config(page_title="Chatbot Rosita", page_icon="🌸", layout="centered")

# CSS personalizado para colores y tipografía
st.markdown("""
    <style>
    /* Fondo rosa suave */
    .main {
        background: linear-gradient(135deg, #ffe6f0 0%, #fff0f5 100%);
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #4b2354;
    }
    /* Encabezados */
    h1 {
        color: #d147a3;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    /* Texto info */
    .stTextInput>div>div>input {
        border-radius: 15px;
        border: 2px solid #d147a3;
        padding: 10px 15px;
        font-size: 1.1rem;
        color: #4b2354;
    }
    /* Botón */
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
    /* Dataframe style */
    .stDataFrame {
        border-radius: 15px;
        border: 2px solid #d147a3;
        padding: 10px;
        margin-top: 15px;
        background: white;
        color: #4b2354 !important;
    }
    /* Respuesta info */
    .stAlert {
        background-color: #fce4ec !important;
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

# --------- APP ---------
st.title("🌸 Chatbot 🌸")

archivo = st.file_uploader("📁 Sube cualquier CSV", type="csv")

if archivo:
    df = pd.read_csv(archivo)
    st.dataframe(df)

    # Concatenamos cada fila en un texto
    filas_texto = df.astype(str).agg(' | '.join, axis=1).tolist()

    # Cargar modelo pequeño para embeddings
    @st.cache_resource(show_spinner=True)
    def cargar_modelo():
        return SentenceTransformer('all-MiniLM-L6-v2')

    model = cargar_modelo()

    # Crear embeddings para filas
    @st.cache_resource(show_spinner=True)
    def crear_embeddings(textos):
        return model.encode(textos, convert_to_tensor=True)

    embeddings_filas = crear_embeddings(filas_texto)

    pregunta = st.text_input("💬 Haz cualquier pregunta sobre los datos:")

    if pregunta:
        emb_pregunta = model.encode(pregunta, convert_to_tensor=True)

        # Calculamos similitud coseno con filas
        similitudes = cosine_similarity(emb_pregunta.reshape(1, -1), embeddings_filas)[0]
        idx_mejor = np.argmax(similitudes)
        mejor_similitud = similitudes[idx_mejor]

        # Umbral mínimo para considerar que la pregunta es relevante
        UMBRAL_SIMILITUD = 0.4

        if mejor_similitud >= UMBRAL_SIMILITUD:
            st.write("✅ Fila más relevante encontrada:")
            st.dataframe(df.iloc[[idx_mejor]])
            st.write(f"🔍 Similitud: {mejor_similitud:.3f}")
        else:
            st.warning("❌ No encontré información relacionada con tu pregunta en el archivo.")
