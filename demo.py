import streamlit as st 

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# page config

st.set_page_config(page_title="C++ RAG chatbot")
st.title("C++ RAG chatbot")
st.write("ask any question related to c++ introduction")

# laod environment variable

@st.cache_resource
def load_vectorstore():
    #load document
    loader=TextLoader("C++_introduction.txt",encoding="utf-8")
    documents=loader.load()
    
    #split text
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20 
        
    )
    final_documents=text_splitter.split_documents(documents)
    
    # Embeddings
    embeddings=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # create faiss vector store
    
    db=FAISS.from_documents(final_documents,embeddings)
    return db

#load vector db(only once)
db=load_vectorstore()

#user input

query=st.text_input("enter your question about C++:")

if query:
    docs=db.similarity_search(query,k=3)
    st.subheader("Retrived Context")
    
    for i, doc in enumerate(docs):
        st.markdown(f"**Result**{i+1}")
        st.write(doc.page_content)


    
    