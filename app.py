from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pdfminer.high_level import extract_text
from langchain.callbacks.tracers import ConsoleCallbackHandler


# Langchain code
def get_OpenAI_response(prompt, input_job_description, input_pdf_text):
    prompt = ChatPromptTemplate.from_template(prompt)
    llm = ChatOpenAI(temperature=0.8).with_config({'callbacks': [ConsoleCallbackHandler()]})
    chain = prompt | llm
    return chain.invoke({"job_description": input_job_description, "pdf_text": input_pdf_text})


prompt = "Please act like an Application Tracking System as experienced recruiter with a deep understanding of the Job profiles Software Developer, Data Science expert," \
         " Machine Learning Engineer, Software Engineer, Data Analyst, Big Data Engineer. Your task is to evaluate the resume based on the given Job Description and" \
         " provide a score in terms of percentage match. Please consider that the kob market is very competitive, therefore also provide the best possible assistance for " \
         "improving the resumes. Assign the percentage match score with job description, and the missing keywords with high accuracy." \
         "resume: {pdf_text}" \
         "job description: {job_description}" \
 \
    # streamlit code
st.title(" Smart Automated Resume Screening")
st.text("Improve your resume screening process by using AI to match job descriptions with resumes")

input_job_description = st.text_area("Enter the job description")
uploaded_file = st.file_uploader("Upload a resume", type=["pdf"], help="Upload a resume in PDF format")

submit_button = st.button("Evaluate")

if submit_button:
    if uploaded_file is not None:
        pdf_text = extract_text(uploaded_file)
        st.title(get_OpenAI_response(prompt, input_job_description, pdf_text).content)
    else:
        st.write("Please upload a resume")
