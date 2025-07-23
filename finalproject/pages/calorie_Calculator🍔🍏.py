from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import time

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Pro Vision API and get response
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to handle image file upload
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize the Streamlit app
st.set_page_config(page_title="🍏 Gemini Health App 🥗")

# Header and Introduction
st.header("🍏 Gemini Health App 🥗")
st.write(
    "Welcome to the **Gemini Health App**! Upload an image of your food, and we'll help you analyze the **calories** in your meal. 🌱"
)
st.write(
    "Simply upload an image, and we will calculate the total calories and provide a detailed breakdown of each food item. 💡"
)

# Food Description Input
st.subheader("🍴 **Describe your food:**")
input = st.text_input(
    "🔍 **Enter a description of the food:**", 
    key="input", 
    placeholder="Enter specific details like 'Salad, chicken, etc.'"
)

# File Upload for Image
st.subheader("📸 **Upload a Food Image**")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image Display with Reduced Size
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ✅ Resize Image for Display (Reduced Size)
    image = image.resize((300, 300), Image.LANCZOS)  # Resize to a smaller display size
    st.image(image, caption="📸 Uploaded Food Image",  use_container_width=False, width=300)  # Adjust width

    # Allow users to download the uploaded image
    st.download_button(
        label="🔽 Download Image",
        data=uploaded_file,
        file_name="uploaded_image.png",
        mime="image/png",
    )

# Progress bar while the image is being processed
if st.button("💥 **Calculate Calories**"):
    if uploaded_file is not None:
        with st.spinner("🧠 Analyzing your food... Please wait..."):
            time.sleep(2)  # Simulate processing time
            image_data = input_image_setup(uploaded_file)

            # Generate the response using the image and input prompt
            input_prompt = """
                You are an expert nutritionist. Based on the food items in the image,
                calculate the total calories and provide the details of every food item
                with its respective calorie intake. Respond in this format:

                1. Item 1 - [no of calories]
                2. Item 2 - [no of calories]
                ----
                ----
            """
            response = get_gemini_response(input_prompt, image_data, input)

            # Display the response
            st.subheader("🍽️ **Calorie Breakdown**")
            st.write(response)

    else:
        st.error("❌ **Please upload an image to calculate calories.**")

# Footer section
st.markdown("""
    ---  
    💬 Have questions? Contact us at [vaishnavikalambate241@gmail.com](mailto:vaishnavikalambate241@gmail.com)  
    
""")

# Sidebar section for additional options
st.sidebar.header("🛠️ **Additional Features**")
st.sidebar.write("Explore some additional features:")
st.sidebar.write("- **Image Enhancement**: Enhance image quality before processing.")
st.sidebar.write("- **Clear Uploaded Image**: Refresh the page to upload a new image.")

st.sidebar.markdown("""
    ---  
    👩‍🍳 **About the App**: This app utilizes AI to analyze your food image and calculate the calories and nutritional breakdown.  
    👩‍💻 **Developed with ❤️ by Health Team**
""")
