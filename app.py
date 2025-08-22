# --- PARCHE PARA SQLITE EN STREAMLIT CLOUD ---
# Debe estar en la parte SUPERIOR del script para que funcione.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------------

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Cargar la clave de API desde los secretos de Streamlit o un archivo .env
load_dotenv()
# Para el despliegue en Streamlit Cloud, usaremos st.secrets
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Para pruebas locales

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
st.set_page_config(page_title="Asistente de Documentación (Groq)", layout="wide")
st.title("⚡ Asistente de Documentación Técnica con Groq")
st.write("Esta herramienta responde preguntas basándose en la documentación interna de los productos a una velocidad increíble.")

# --- LÓGICA DE INGESTA Y CACHÉ ---
@st.cache_resource
def cargar_y_procesar_datos():
    with st.spinner("Cargando y procesando documentos... Este proceso solo se ejecuta una vez."):
        docs = []
        for file_path in os.listdir("documentos"):
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join("documentos", file_path))
                docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Usar embeddings gratuitos y de código abierto de Hugging Face
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Crear la base de datos vectorial en memoria
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever(search_kwargs={'k': 5}) # Aumentamos a 5 fragmentos para dar más contexto

# --- INICIALIZACIÓN DEL MODELO Y LA CADENA ---
try:
    retriever = cargar_y_procesar_datos()
    
    # Usar el modelo de lenguaje de Groq (Llama 3 8B)
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    
    template = """
    Eres un asistente experto en la documentación técnica de nuestros productos.
    Tu tarea es responder a las preguntas de los empleados basándote únicamente en el siguiente contexto.
    Sé conciso y directo. Si la respuesta no se encuentra en el contexto, indica claramente: "No tengo suficiente información en la documentación para responder a esa pregunta."
    
    Contexto:
    {context}
    
    Pregunta:
    {input}
    
    Respuesta útil:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

except Exception as e:
    st.error(f"Error al inicializar la aplicación: {e}")
    st.info("Asegúrate de haber configurado tu clave de API de Groq correctamente.")
    st.stop()

# --- INTERFAZ DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("¿Qué información técnica necesitas?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Se añade un generador para simular el streaming de la respuesta de Groq
        response_generator = retrieval_chain.stream({"input": user_query})
        st.write_stream(part["answer"] for part in response_generator if "answer" in part)
    
    # Se reconstruye la respuesta completa para guardarla en el historial
    full_response = "".join([part["answer"] for part in retrieval_chain.stream({"input": user_query}) if "answer" in part])
    st.session_state.messages.append({"role": "assistant", "content": full_response})
