__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

# load .env
load_dotenv()


import streamlit as st
import tempfile
import os


# title
st.title("ChatCSV")
st.write("---")


# upload
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type="csv")
st.write("---")


def csv_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = CSVLoader(temp_filepath)
    docs = loader.load()
    return docs


# after upload
if uploaded_file is not None:
    docs = csv_to_document(uploaded_file)

    # Splitter
    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0,
    )
    texts = text_splitter.split_documents(docs)



    # Embedding
    embeddings_model = OpenAIEmbeddings()


    # Chroma
    db = Chroma.from_documents(texts, embeddings_model)


    # Generate Q&A
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요.")
    
    if st.button("질문하기"):
        with st.spinner("답변 생성중..."):        
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])