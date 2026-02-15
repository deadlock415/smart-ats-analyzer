import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

# 1. Setup the Local LLM (Ollama)
# We use 'llama3' or 'mistral'. Make sure she has pulled one of them.
llm = ChatOllama(model="llama3", format="json", temperature=0)

# 2. Define the "Recruiter Persona" Prompt
prompt = PromptTemplate(
    template="""
    You are an expert Technical Recruiter and ATS (Application Tracking System). 
    Your job is to evaluate a candidate's resume against a provided Job Description (JD).

    ### JOB DESCRIPTION:
    {jd_text}

    ### RESUME TEXT:
    {resume_text}

    ### INSTRUCTION:
    Compare the resume against the JD. Look for keywords, skills, and experience alignment.
    Return a valid JSON object with the following keys:
    - "match_score": (integer between 0-100)
    - "missing_keywords": (list of specific technical skills present in JD but missing in Resume)
    - "profile_summary": (a 2-sentence professional summary tailored to this JD)
    - "advice": (1 specific tip to improve the resume)

    JSON OUTPUT:
    """,
    input_variables=["jd_text", "resume_text"]
)

# 3. Create the Chain
chain = prompt | llm | JsonOutputParser()

def process_resume(pdf_path, jd_text):
    """
    Loads PDF, extracts text, and runs the LLM chain.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # Combine text from all pages
        resume_text = "\n".join([doc.page_content for doc in docs])
        
        # Run the Chain
        result = chain.invoke({"jd_text": jd_text, "resume_text": resume_text})
        return result
    except Exception as e:
        return {"error": str(e)}