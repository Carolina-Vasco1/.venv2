from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
import langchain
from datetime import datetime
from transformers import pipeline

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

st.markdown(
    """
    <style>
        body {
            background-color: #8a2be2;  /* Cambiar el color */
        }
    </style>
    """,
    unsafe_allow_html=True
)
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    knowledge_base = Chroma.from_texts(chunks, embeddings)

    return knowledge_base

# Función principal de la aplicación
def main():
    st.title("Preguntas a un Documento")

    # Opción para cargar un archivo PDF o agregar texto
    option = st.radio("Seleccione una opción:", ["Cargar PDF", "Agregar Texto"])

    if option == "Cargar PDF":
        pdf = st.file_uploader("Sube tu archivo PDF", type="pdf")

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()

            knowledgeBase = process_text(text)

            # Caja de entrada de texto para que el usuario escriba su pregunta
            query = st.text_input('Escribe tu pregunta para el PDF...')

            # Botón para cancelar la pregunta
            cancel_button = st.button('Cancelar')

            if cancel_button:
                st.stop()  # Detiene la ejecución de la aplicación

            if query:
                # Realiza una búsqueda de similitud en la base de conocimientos
                docs = knowledgeBase.similarity_search(query)

                # Inicializa un modelo de lenguaje de OpenAI y ajusta sus parámetros
                model = "gpt-3.5-turbo-instruct"  # Acepta 4096 tokens
                temperature = 0  # Valores entre 0 - 1

                llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

                # Carga la cadena de preguntas y respuestas
                chain = load_qa_chain(llm, chain_type="stuff")

                # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
                with get_openai_callback() as cost:
                    start_time = datetime.now()
                    response = chain.invoke(input={"question": query, "input_documents": docs})
                    end_time = datetime.now()
                    print(cost)  # Imprime el costo de la operación

                    st.write(response[
                                 "output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación

                    # Muestra el número de tokens consumidos y el tiempo de transacción
                    if 'usage' in response:
                        st.write(f"Tokens consumidos: {response['usage']['total_tokens']}")
                        st.write(f"Tokens de solicitud: {response['usage']['prompt_tokens']}")
                        st.write(f"Tokens de finalización: {response['usage']['completion_tokens']}")
                    else:
                        st.write("No se pudo obtener la información de uso de la respuesta de la API.")
                        st.write(f"Tiempo de transacción: {end_time - start_time}")

    else:
        def classify_text_with_zero_shot(text):
            classifier = pipeline("zero-shot-classification")
            candidate_labels = ["política", "deporte", "religión", "otro"]

            result = classifier(text, candidate_labels)

            return result["labels"][0]

        # agregar texto manualmente
        st.subheader("Agregar Texto Manualmente:")
        input_text = st.text_area("Escribe o pega tu texto aquí:")

        if not input_text:
            st.warning("Por favor, ingresa texto antes de continuar.")
            st.stop()
            
# clasifica el input text into categories using Zero-Shot Classification
        category = classify_text_with_zero_shot(input_text)
        st.write(f"Texto clasificado como: {category}")

            

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()  # Llama a la función principal
