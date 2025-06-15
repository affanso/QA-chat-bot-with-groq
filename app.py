from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{input}

"""
)

def gen_response(input):
    response = st.session_state.rag_chain.invoke({"input":input})
    return response



def create_vector_embedding(uploaded_file):
    
    llm = ChatGroq(model="gemma2-9b-it")

    embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    st.session_state.chunked_docs = splitter.split_documents(docs)
    st.session_state.vectordb = Chroma.from_documents(documents=st.session_state.chunked_docs,embedding=embedding_model)
    
    st.session_state.retriever = st.session_state.vectordb.as_retriever()
    st.session_state.question_answer_chain=create_stuff_documents_chain(llm,prompt)
    st.session_state.rag_chain=create_retrieval_chain(st.session_state.retriever,st.session_state.question_answer_chain)


#Streamlit APP
st.title("QA Chat Bot With Groq")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    st.write("File name:", uploaded_file.name)
    

if st.button("Document Embedding"):
    if uploaded_file:
        create_vector_embedding(uploaded_file)
        st.write("Vector Database is ready")
    elif not uploaded_file:
        st.write("Please Upload the file first")



st.write("Go ahead and ask any question about the pdf")
input = st.text_input("You:")


if input and uploaded_file:
    response = gen_response(input)
    st.write(response['answer'])
    # 23 #With a streamlit expander
    with st.expander ("Document similarity Search"):
        for i, doc in enumerate (response['context']):
            st.write(doc.page_content)
            st.write('-----------------------------------')
elif not input:
    st.write("Please provide the query")
elif not uploaded_file:
    st.write("Please Upload the file first")
