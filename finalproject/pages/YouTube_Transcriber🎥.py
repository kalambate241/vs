import streamlit as st
import os
import re
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import yt_dlp

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract YouTube video ID
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# ✅ Fetch transcript using yt_dlp
def fetch_transcript_ytdlp(video_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writesubtitles': True,
        'subtitlesformat': 'json3',
        'writeautomaticsub': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get('subtitles') or info.get('automatic_captions')

            if not subtitles:
                st.error("⚠️ No subtitles or captions available for this video.")
                return None

            # Try English subtitles first
            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subtitles:
                    sub_url = subtitles[lang][0]['url']
                    response = requests.get(sub_url)
                    if response.ok:
                        json_data = response.json()
                        return " ".join([e['segs'][0]['utf8'] for e in json_data['events'] if 'segs' in e])
            st.error("⚠️ English captions not found.")
            return None
    except Exception as e:
        st.error(f"❌ yt-dlp transcript fetch failed: {e}")
        return None

# ✅ Generate content using Gemini based on summary length
def generate_gemini_content(transcript_text, summary_length):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        if summary_length == "Short":
            custom_prompt = (
                "Summarize the following video transcript in 3-4 short bullet points (under 100 words):\n\n"
            )
        elif summary_length == "Medium":
            custom_prompt = (
                "Summarize the following video transcript in 5–7 bullet points with moderate detail (150–200 words):\n\n"
            )
        elif summary_length == "Long":
            custom_prompt = (
                "Provide a detailed summary of the following video transcript in 10+ bullet points with rich explanations (around 300 words):\n\n"
            )
        else:
            custom_prompt = "Summarize the following transcript:\n\n"

        response = model.generate_content(custom_prompt + transcript_text)
        return response.text

    except Exception as e:
        st.error(f"❌ Error generating summary: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Transcript to Summary", layout="wide")
st.title("🎥 YouTube Transcript → AI Summary")
st.markdown("Paste any YouTube link to generate a clean summary using Google Gemini ✨")

youtube_link = st.text_input("Enter YouTube Video URL:", placeholder="e.g. https://www.youtube.com/watch?v=5MgBikgcWnY")

if youtube_link:
    video_id = extract_video_id(youtube_link)
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", caption="Video Thumbnail", width=300)

summary_length = st.selectbox("Select Summary Length", ["Short", "Medium", "Long"], index=1)

if st.button("🔍 Generate Summary"):
    if youtube_link:
        with st.spinner("⏳ Extracting transcript and summarizing..."):
            transcript_text = fetch_transcript_ytdlp(youtube_link)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, summary_length)
                if summary:
                    st.markdown("## 📄 AI Summary:")
                    st.write(summary)
                    st.markdown(f"### 📝 Word Count: {len(summary.split())} words")

                    st.download_button(
                        label="💾 Download Summary",
                        data=summary,
                        file_name="youtube_summary.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("⚠️ Failed to fetch transcript. Try another video.")
    else:
        st.warning("⚠️ Please enter a valid YouTube video URL.")

if st.button("🧹 Clear"):
    st.experimental_rerun()

# Optional styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #0066CC;
        color: white;
        font-size: 16px;
        border-radius: 6px;
    }
    .stTextInput>div>input {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)
