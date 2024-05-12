import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    """
"""
    text = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        filename = pdf.name
        text[filename] = ""
        for page in pdf_reader.pages:
            text[filename] += page.extract_text()
    return text

def get_text_chunks(text):
    """
"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks_list = []
    for key, value in text.items():
        chunks = text_splitter.split_text(value)
        for item in chunks:
            chunks_list.append({"Nome_Arquivo":key, "Chunk":item})
    return chunks_list

def handle_userinput(user_question,model):
    """
"""
    resposta = gerar_e_buscar_consulta(user_question,st.session_state.df_embed,model)
    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(resposta)
    st.session_state.chat_place.empty()
    with st.session_state.chat_place.container():
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

def embed_fn(model,title, text):
    return genai.embed_content(model=model,
                                 content=text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")["embedding"]

def gerar_e_buscar_consulta(consulta, base, model):
    embedding_da_consulta = genai.embed_content(model=model,
                                 content=consulta,
                                 task_type="RETRIEVAL_QUERY")["embedding"]

    produtos_escalares = np.dot(np.stack(base["Embeddings"]), embedding_da_consulta)
    indices = np.argsort(produtos_escalares)[-5:]
    df = base.iloc[indices]["Chunk"]
    generative_model = genai.GenerativeModel('gemini-1.0-pro')
    context = """Baseado no Contexto abaixo responda a minha pergunta\n\n"""
    for item in df:
        context += item
        context += "\n\n"
    chat = generative_model.start_chat(history=[])
    response = chat.send_message(context + f"Pergunta: {consulta}")
    return response.text

def main():
    """
"""
    load_dotenv()
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    embedding_model = "models/embedding-001"
    st.set_page_config(page_title="Converse com vários PDFs",
                       page_icon=":books:")

    st.write(css, unsafe_allow_html=True)
    
    if "df_embed" not in st.session_state:
        st.session_state.df_embed = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ["Olá Gemini","Olá Humano"]
    if "chat_place" not in st.session_state:
        st.session_state.chat_place = None
    
    st.header("Converse com vários PDFs :books:")
    st.session_state.chat_place = st.empty()
    user_question = st.text_input("Faça perguntas sobre os seus documentos:")
    if user_question:
        handle_userinput(user_question,embedding_model)
    
    with st.session_state.chat_place.container():
        st.write(user_template.replace("{{MSG}}", "Olá Gemini"), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", "Olá Humano"), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Seus Documentos")
        pdf_docs = st.file_uploader(
            "Carregue seus PDFs aqui e clique em 'Processar'", accept_multiple_files=True)
        if st.button("Processar"):
            with st.spinner("Processando"):
                # Carregar os textos dos arquivos pdf
                raw_text = get_pdf_text(pdf_docs)

                # Pegar pedaços dos textos para facilitar a busca semântica
                text_chunks = get_text_chunks(raw_text)
                df = pd.DataFrame(text_chunks)
                df["Embeddings"] = df.apply(lambda row: embed_fn(embedding_model,row["Nome_Arquivo"], row["Chunk"]), axis=1)
                st.session_state.df_embed = df

if __name__ == '__main__':
    main()