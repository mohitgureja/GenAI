import os
from langchain.callbacks.tracers import ConsoleCallbackHandler
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Streamlit code
st.title("LangChain")
input_text = st.text_input("Enter the query to the thesis document")

# LangChain code
embeddings = OpenAIEmbeddings()

# Load the documents
loader = PyPDFLoader("examples/Automatic_Generation_of_a_Custom_Corpora_for_Invoice_Analysis_and_Recognition.pdf")
pages = loader.load()

# Load the vectorstore
faiss_index = FAISS.from_documents(pages, embeddings)

llm = ChatOpenAI(temperature=0.6).with_config({'callbacks': [ConsoleCallbackHandler()]})
prompt = ChatPromptTemplate.from_template("""Answer the question based on this provided context:"
                                          <context>
                                          {context}
                                          </context>
                                        Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = faiss_index.as_retriever()

# Create the retrieval chain
retriever_chain = create_retrieval_chain(retriever, document_chain)

if input_text:
    st.write(retriever_chain.invoke({"input": input_text})['answer'])
