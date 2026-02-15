import streamlit as st
import os
from chains import process_resume

# Page Config
st.set_page_config(page_title="Smart ATS Analyzer", layout="wide")

st.title("🚀 GenAI Smart Resume Analyzer")
st.markdown("### Built with LangChain & Llama 3 (Local Privacy)")

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Paste Job Description (JD)")
    jd_text = st.text_area("Paste the JD here...", height=300)

with col2:
    st.subheader("2. Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Upload your Resume", type="pdf")

# Action Button
if st.button("Analyze Resume"):
    if not jd_text or not uploaded_file:
        st.warning("Please provide both a JD and a Resume PDF!")
    else:
        with st.spinner("🤖 AI is reading your resume... (This runs locally on Ollama)"):
            # Save uploaded file temporarily so PyPDFLoader can read it
            temp_path = os.path.join("data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Call the LangChain Logic
            response = process_resume(temp_path, jd_text)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Display Results
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.success("Analysis Complete!")
                
                # Metrics
                score = response.get("match_score", 0)
                st.metric(label="ATS Match Score", value=f"{score}%")
                
                # Progress Bar
                st.progress(score / 100)
                
                # Details
                st.subheader("🔍 Missing Keywords")
                st.write(response.get("missing_keywords", []))
                
                st.subheader("📝 Tailored Profile Summary")
                st.info(response.get("profile_summary", "No summary generated."))
                
                st.subheader("💡 Expert Advice")
                st.write(response.get("advice", "No advice generated."))