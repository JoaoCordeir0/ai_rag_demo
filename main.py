import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURAÇÃO DA API ---'
GOOGLE_API_KEY = "..."
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- TÍTULO ---
st.title("🤖 AI with RAG + PDF/Excel")

# --- UPLOAD DO ARQUIVO ---
uploaded_file = st.file_uploader("📄 Envie um PDF ou Excel (.xlsx)", type=["pdf", "xlsx"])

# --- PROCESSAMENTO DO ARQUIVO ---
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

    elif file_ext == "xlsx":
        df = pd.read_excel(uploaded_file, skiprows=2)
        st.write(df.columns)
        df = df[df['NF USINA'].notna()]
        df['OBSERVAÇÃO'] = df['OBSERVAÇÃO'].fillna('')
        loader = DataFrameLoader(df, page_content_column="OBSERVAÇÃO")
        docs = loader.load()

    else:
        st.error("Formato de arquivo não suportado.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = FAISS.from_documents(docs_chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    user_question = st.text_input("❓ Pergunte algo com base no conteúdo do arquivo:")

    if user_question:
        with st.spinner("🔎 Buscando resposta..."):
            resposta = qa_chain.run(user_question)
            st.success(resposta)