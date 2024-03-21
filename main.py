"""
# My first app
Here's our first attempt at using data to create a table:
"""

import os
import base64
import tempfile
import streamlit as st
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY = st.secrets["openai_api_key"]
OPENAI_BASE_URL = st.secrets["openai_base_url"]
os.environ['OPENAI_API_KEY'] =OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] =OPENAI_BASE_URL

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,openai_api_base=OPENAI_BASE_URL)
persist_directory='pdf_persist'
collection_name='pdf_collection'

llm=ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY,openai_api_base=OPENAI_BASE_URL,model_name="gpt-4-1106-preview")
chain=load_qa_chain(llm,chain_type='stuff')

def load_pdf(pdf_path):
    docs=PyMuPDFLoader(pdf_path).load()
    # 将pdf向量化
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(split_docs, embeddings,
                                        collection_name=collection_name,
                                        persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

def show_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        fp = Path(tmp_file.name)
        fp.write_bytes(uploaded_file.getvalue())
        with open(tmp_file.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
                      f'width="800" height="1000" type="application/pdf">'

    return pdf_display

st.set_page_config(layout="wide")


col1, col2 = st.columns([0.5, 0.5])
with col1:
    st.header("上传PDF",divider="grey")
    with st.container():
        uploaded_file=st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None and (not st.session_state.get("uploaded_file") or uploaded_file.name != st.session_state.uploaded_file.get("name")):
            # Update the session state
            st.session_state["uploaded_file"] = {"name": uploaded_file.name, "data": uploaded_file}
            path=os.path.join('.',uploaded_file.name)
            with open(path,'wb') as f:
                f.write(uploaded_file.getbuffer())

            current_vectorstore = load_pdf(path)
            st.session_state["current_vectorstore"]=current_vectorstore
            os.unlink(uploaded_file.name)

    with st.container():
        on = st.toggle('显示上传的文件')
        if on and "uploaded_file" in st.session_state and st.session_state["uploaded_file"].get("data") is not None:
            pdf_display = show_pdf(uploaded_file)
            st.markdown(pdf_display, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with col2:
    # chat=ChatOpenAI(openai_api_key=OPENAI_API_KEY,openai_api_base=OPENAI_BASE_URL)
    st.header("PDF解读",divider="grey")

    st.subheader("当前问答")
    prompt=st.chat_input("输入问题")
    if prompt and prompt is not None and prompt != "":
        if "current_vectorstore" in st.session_state:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state["messages"].append(['user', prompt])
            current_vectorstore=st.session_state['current_vectorstore']
            docs = current_vectorstore.similarity_search(prompt, 3)
            answer=chain.run(input_documents=docs,question=prompt)
            st.session_state["messages"].append(['assistant',answer])
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            st.write("请先上传pdf")


    st.subheader("历史聊天记录")
    with st.container(height=500):
        for message in st.session_state["messages"]:
            with st.chat_message(message[0]):
                st.markdown(message[1])

