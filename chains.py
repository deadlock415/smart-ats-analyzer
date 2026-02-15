import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

# 1. Setup Local LLM (Keep temperature 0 for consistency)
llm = ChatOllama(model="llama3", format="json", temperature=0)

# 2. The "Ruthless ATS" Prompt
# We force the model to think in steps: Extract JD Skills -> Check Resume -> Calculate Score
prompt = PromptTemplate(
    template="""
    You are a strict Application Tracking System (ATS) used by tech giants. 
    Your job is to objectively calculate the match percentage of a resume to a job description based on KEYWORDS.

    ### JOB DESCRIPTION (JD):
    {jd_text}

    ### RESUME:
    {resume_text}

    ### INSTRUCTION:
    1. Extract the top 15 most critical technical skills/keywords from the JD (e.g., Python, RAG, AWS, Docker, NLP, SQL, etc.).
    2. Check if each exact keyword is present in the Resume.
    3. Calculate the Match Score = (Number of Matched Keywords / Total Critical Keywords) * 100.
    4. List ALL missing keywords from that critical list.

    ### OUTPUT FORMAT (JSON ONLY):
    {{
        "match_score": <integer_0_to_100>,
        "missing_keywords": ["keyword1", "keyword2", "keyword3"],
        "advice": "One brutal sentence on how to fix the resume."
    }}
    """,
    input_variables=["jd_text", "resume_text"]
)

# 3. Create the Chain
chain = prompt | llm | JsonOutputParser()

def process_resume(pdf_path, jd_text):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        resume_text = "\n".join([doc.page_content for doc in docs])
        
        # Run the Chain
        return chain.invoke({"jd_text": jd_text, "resume_text": resume_text})
    except Exception as e:
        return {"error": str(e)}