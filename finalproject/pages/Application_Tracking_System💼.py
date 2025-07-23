import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import re
from textstat import flesch_reading_ease

# Load environment variables from .env
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get AI response from Gemini
def get_gemini_response(resume_text, job_desc):
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Prompt AI to return only JSON
    prompt = f"""
    You are an AI-powered ATS (Applicant Tracking System) specialized in resume screening. 
    Analyze the given resume against the job description and return **ONLY a valid JSON response**.

    **Output Format (STRICTLY return only JSON, no extra text):**
    {{
        "JD Match": "75%",
        "MissingKeywords": ["Python", "Machine Learning", "SQL"],
        "Profile Summary": "The resume aligns well with the job description but lacks SQL and ML experience."
    }}

    **Resume:** {json.dumps(resume_text)}
    **Job Description:** {json.dumps(job_desc)}

    **Return ONLY a valid JSON (no extra words or explanations).**
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Attempt direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Use regex to extract JSON part if AI adds extra text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        # If all fails, return error
        return {"error": "Invalid AI response format"}
    
    except Exception as e:
        return {"error": str(e)}

# Function to extract text from the uploaded PDF
def extract_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text() or ""
    return text.strip()

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to calculate readability score
def calculate_readability_score(text):
    return flesch_reading_ease(text)

# Streamlit App UI
st.set_page_config(page_title="Smart ATS", layout="wide")
st.title("💼 **Smart ATS: Resume Enhancer**")
st.text("Boost Your Resume's ATS Compatibility 🚀")

# Job description input
jd = st.text_area("📄 Paste the Job Description (JD):", height=150, placeholder="Enter the job description here...")

# Resume upload
uploaded_file = st.file_uploader("📎 Upload Your Resume (PDF)", type=["pdf"], help="Please upload your resume in PDF format.")

# Submit button
submit = st.button("🔍 Analyze Resume", use_container_width=True)

# When the submit button is pressed
if submit:
    if uploaded_file is not None and jd.strip():
        with st.spinner('🔄 Processing your resume and job description...'):
            # Extract text from the uploaded resume
            resume_text = extract_pdf_text(uploaded_file)

            # Get AI analysis from Gemini
            response = get_gemini_response(resume_text, jd)

            # Check if response is valid
            if "error" in response:
                st.error(f"⚠️ AI Error: {response['error']}")
            else:
                jd_match = response.get("JD Match", "N/A")
                missing_keywords = response.get("MissingKeywords", [])
                profile_summary = response.get("Profile Summary", "N/A")

                # Display JD Match Percentage
                st.subheader("✅ **ATS Evaluation Result:**")
                st.markdown(f"### **JD Match:** {jd_match}")

                # Display as a progress bar
                try:
                    jd_match_percentage = float(jd_match.replace('%', '').strip())
                    st.progress(jd_match_percentage / 100)
                except ValueError:
                    st.error("❌ Invalid JD match percentage format.")

                # Display missing keywords
                st.markdown("### **🔑 Missing Keywords:**")
                if missing_keywords:
                    st.write(", ".join(missing_keywords))
                else:
                    st.write("✔️ No missing keywords found.")

                # Display profile summary
                st.markdown("### **📌 Profile Summary:**")
                st.write(profile_summary)

                # Display Readability Score
                st.subheader("📊 **ATS Readability Score:**")
                st.markdown(f"**Resume Readability Score:** {calculate_readability_score(resume_text)}")
                st.markdown(f"**Job Description Readability Score:** {calculate_readability_score(jd)}")

                # Display WordClouds for JD and Resume Text
                st.subheader("📈 **Keyword Visualization**")
                st.markdown("### **Job Description Keywords:**")
                st.pyplot(generate_wordcloud(jd))

                st.markdown("### **Resume Keywords:**")
                st.pyplot(generate_wordcloud(resume_text))

                # Improvement Suggestions
                st.subheader("💡 **Suggestions for Improvement:**")
                if missing_keywords:
                    st.markdown("Consider adding these keywords to improve your resume:")
                    st.write(", ".join(missing_keywords))
                else:
                    st.write("✔️ Your resume is well-optimized for the job description.")

                # Allow downloading the analysis results
                analysis_result = json.dumps(response, indent=4)
                buffer = io.BytesIO()
                buffer.write(analysis_result.encode())
                buffer.seek(0)
                st.download_button("📥 Download Analysis", buffer, file_name="ATS_Analysis_Result.json")

    else:
        st.error("❌ Please upload a resume and provide a job description.")

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Customize your resume based on the job description for better ATS ranking.")
st.markdown("📩 Need help? Contact us at [vaishnavikalambate241@gmail.com](mailto:vaishnavikalambate241@gmail.com).")
