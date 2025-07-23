import streamlit as st

# Set page configuration with an attractive layout
st.set_page_config(page_title="AI Multi-Tool Suite", layout="wide")

# Main Title and Subtitle
st.title("🌟 AI Multi-Tool Suite")
st.markdown(
    "<h4 style='color:#4CAF50;'>Your AI-powered assistant for enhanced productivity 🚀</h4>",
    unsafe_allow_html=True,
)
st.write("Welcome to the AI-powered multipage application. Select a tool from the sidebar to begin.")

# Sidebar message
st.sidebar.success("🎯 Select a page above to start your journey!")

# Divider for neat structure
st.markdown("---")

# 🔥 Feature Highlights Section
st.markdown("## 🌟 Available AI Tools")
st.write("Explore the powerful AI-driven applications available in this suite:")

# 📄 Smart ATS Tool
st.markdown("### 💼 **Smart ATS (Resume Analyzer)**")
st.markdown(
    """
    - 📌 **Optimize your resume** for Applicant Tracking Systems (ATS).
    - 🔍 **Match your resume** with a Job Description to see keyword relevance.
    - 📊 **Get a percentage match score** & missing keyword analysis.
    """
)

# 🧾 Invoice Extractor
st.markdown("### 📄 **Invoice Extractor (AI Document Scanner)**")
st.markdown(
    """
    - 📜 **Extract key details** from invoice images.
    - 🔍 **Automated AI-powered scanning** for faster processing.
    - 📂 **Supports various formats (JPG, PNG, PDF).**
    """
)

# 🎥 YouTube Transcriber
st.markdown("### 🎥 **YouTube Transcriber (AI Video Summarizer)**")
st.markdown(
    """
    - 🎙️ **Extract & summarize transcripts** from YouTube videos.
    - 📑 **Get detailed summaries** in bullet points.
    - 🚀 **Perfect for students, researchers, and professionals.**
    """
)

# ✂️ Background Remover Tool
st.markdown("### ✂️ **Background Remover**")
st.markdown(
    """
    - 🖼️ **Remove the background** from images instantly.
    - 🎨 **Keep only the subject** of your image with high accuracy.
    - 🌐 **Supports PNG, JPG, and other image formats**.
    """
)

# 🍎 Calorie Calculator from Image
st.markdown("### 🍏 **Calorie Calculator (AI Image-Based)**")
st.markdown(
    """
    - 🍔 **Upload an image** of food and calculate its calories.
    - 🧠 **AI-powered recognition** to analyze food items.
    - 📈 **Get an accurate calorie count** for various food items.
    """
)

# 📄 Chat with PDF Tool
st.markdown("### 📚 **Chat with PDF (Multiple Files Support)**")
st.markdown(
    """
    - 🗂️ **Upload and analyze multiple PDFs** at once.
    - 💬 **Ask questions** based on the content of the PDFs.
    - 🧠 **Advanced AI** interprets the text to provide relevant answers.
    """
)

# 📊 Data Explorer
st.markdown("### 📊 **Data Explorer**")
st.markdown(
    """
    - 📈 **Upload datasets** and visualize trends effortlessly.
    - 🔍 **Generate insights** with AI-powered analysis.
    - 📊 **Supports CSV, Excel, and other data formats**.
    """
)


# 😊 Sentiment Analyzer
st.markdown("### 😊 **Sentiment Analyzer**")
st.markdown(
    """
    - 💬 **Analyze text sentiment** for positive, neutral, or negative tones.
    - 📊 **AI-powered insights** for customer feedback, social media, and more.
    - 🚀 **Instant sentiment results** with detailed breakdowns.
    """
)

# 📌 How to Use This Application
st.markdown("---")
st.markdown("## 📌 How to Use This AI Suite")
st.markdown(
    """
    1. **Select a tool from the sidebar** 📌.
    2. **Follow the on-screen instructions** for each AI tool.
    3. **Upload necessary files** (PDFs, Images, etc.).
    4. **Let AI process your input & generate insights** 🚀.
    5. **Download or use the AI-generated results**.
    """
)

# 📞 Contact & Support
st.markdown("---")
st.markdown("### 📞 Need Help?")
st.markdown(
    """
    💬 If you have any questions or need support, feel free to reach out:
    - 📧 **Email:** vaishnavikalambate241@gmail.com
    - 🛠️ **FAQ Section in Sidebar**
    
    """
)

# 🏁 Closing Remark
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>Empowering You with AI 🚀</h4>", unsafe_allow_html=True)