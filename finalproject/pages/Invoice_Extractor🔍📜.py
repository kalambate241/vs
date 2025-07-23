from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image, ImageEnhance
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini 1.5 Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to process the response from Gemini
def get_gemini_response(input, image, user_prompt):
    response = model.generate_content([input, image[0], user_prompt])
    return response.text

# Function to extract image details
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# ✅ Function to enhance the uploaded image (Fix for "image has wrong mode" error)
def enhance_image(image):
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")  # Convert image to RGB mode if not already
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)  # Increase contrast by 1.5x

# Function to crop the uploaded image (simple fixed crop)
def crop_image(image):
    left, top, right, bottom = 100, 100, image.width - 100, image.height - 100
    return image.crop((left, top, right, bottom))

# Initialize Streamlit app
st.set_page_config(page_title="📜 MultiLanguage Invoice Extractor", layout="wide")

# Sidebar Information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Iconic_image.png", width=300)
    st.header("📌 About the App")
    st.write("Extract invoice details using AI. Upload an invoice image and ask questions!")
    st.markdown("**Steps:**")
    st.write("1️⃣ Upload an invoice image (JPG, PNG, or JPEG)")
    st.write("2️⃣ Type a question (e.g., What is the total amount?)")
    st.write("3️⃣ Click the button & get AI-generated answers!")

# Header Section
st.header("📜 MultiLanguage Invoice Extractor")

# User Input for Question
input_question = st.text_input(
    "Ask a question about the invoice:", 
    key="input", 
    placeholder="e.g., What is the total amount?",
    help="Type any question about the invoice details."
)

# Image Upload
uploaded_file = st.file_uploader("📂 Upload an Invoice Image", type=["jpg", "jpeg", "png"])

# Image Processing Options
if uploaded_file is not None:
    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display uploaded image with reduced size
    st.image(image, caption="📷 Uploaded Image", width=500)  # Smaller image size

    # Show file size warning if too large
    image_size_mb = uploaded_file.size / (1024 * 1024)  # Convert to MB
    if image_size_mb > 5:
        st.warning("⚠️ The uploaded image is large. It might take longer to process.")

    # Enhance Image Option
    if st.checkbox("✨ Enhance Image"):
        enhanced_image = enhance_image(image)
        st.image(enhanced_image, caption="✨ Enhanced Image", width=500)

    # Crop Image Option
    if st.checkbox("✂️ Crop Image"):
        cropped_image = crop_image(image)
        st.image(cropped_image, caption="✂️ Cropped Image", width=500)

# Store Previous Questions
if "history" not in st.session_state:
    st.session_state["history"] = []

# Display Previous Questions
st.subheader("🔄 Previous Questions")
if st.session_state["history"]:
    for idx, question in enumerate(st.session_state["history"]):
        st.write(f"{idx+1}. {question}")

# Button for Processing Invoice
if st.button("🧠 Analyze Invoice"):
    if uploaded_file is not None:
        # Store question in history
        st.session_state["history"].append(input_question)

        # Process Image & Get AI Response
        image_data = input_image_details(uploaded_file)
        input_prompt = """
            You are an expert in understanding invoices. Based on the uploaded image,
            answer any questions related to the invoice details.
        """
        response = get_gemini_response(input_prompt, image_data, input_question)

        # Display Response
        st.subheader("📚 AI Response:")
        if response:
            st.write(response)

            # Option to Download Response
            st.download_button(
                label="📥 Download Response",
                data=response.encode("utf-8"),
                file_name="invoice_response.txt",
                mime="text/plain",
            )
        else:
            st.error("❌ No response received. Try again with a different question.")
    else:
        st.error("❌ Please upload an invoice image.")

# Styling Enhancements
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .streamlit-expanderHeader {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Help Section (Expandable)
with st.expander("❓ How to Use"):
    st.write("""
    1️⃣ **Upload** an invoice image (JPG, PNG, or JPEG).  
    2️⃣ **Type** a question about the invoice details.  
    3️⃣ **Click the button** to analyze and get AI-generated responses.  
    4️⃣ **Enhance or Crop** the image before analysis if needed.  
    5️⃣ **Download the response** for future reference.  
    6️⃣ **Check previous questions** in the "Previous Questions" section.  
    """)

# Explanation of AI Usage
st.markdown("""
    ## 🔍 How It Works:

    - You can ask about the total amount, invoice date, vendor name, and more.  
""")
