from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant reply the user question"),
        ("user", "{input}")
    ]
)

def gen_response(GROQ_API_KEY,model,input):
    llm = ChatGroq(model=model,api_key=GROQ_API_KEY)
    parser = StrOutputParser()
    chain = prompt|llm|parser
    response = chain.invoke({"input":input})
    return response








st.title("QA Chat Bot With Groq")


st.sidebar.title("Settings")

GROQ_API_KEY = st.sidebar.text_input("Enter your Groq API:",type="password")

model = st.sidebar.selectbox(
    "Choose a model:",
    [
        "gemma2-9b-it",
        "meta-llama/llama-guard-4-12b", 
        "llama-3.3-70b-versatile", 
        "llama-3.1-8b-instant",
    ]
)


st.write("Go ahead and ask any question")
input = st.text_input("You:")

if input and GROQ_API_KEY and model:
    answer = gen_response(GROQ_API_KEY,model,input)
    st.write(answer)
elif not input:
    st.write("Please provide the query")
elif not GROQ_API_KEY:
    st.write("Please provide the API KEY")
elif not model:
    st.write("Please select the model")
