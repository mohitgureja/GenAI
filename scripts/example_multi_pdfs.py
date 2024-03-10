import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

def extract_pdf_text(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        raw_text += extract_text(pdf)
    return raw_text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if you think answer is not in the provided context just say, "Answer is not available in the context", please do not provide the wrong answer \n\n
    Context: {context}
    Question: {question}
    """

    llm = ChatOpenAI(temperature=0.6).with_config({'callbacks': [ConsoleCallbackHandler()]})
    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain


def invoke_chain(user_question):
    embeddings = OpenAIEmbeddings()
    new_vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_vector_db.similarity_search(user_question)
    print(len(docs))
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    st.write(response['output_text'])


def main():
    st.set_page_config("Ask a question from the provided documents")
    st.header("Chat with pdfs using LangChain and GPT3")

    user_question = st.text_input("Ask a question from the provided PDFs")

    if user_question:
        invoke_chain(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDFs and click on Submit Button", type=["pdf"], accept_multiple_files=True)
        submit_button = st.button("Submit")
        if submit_button:
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Done")


if __name__ == "__main__":
    main()
