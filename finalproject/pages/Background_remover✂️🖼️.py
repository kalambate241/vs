import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import zipfile
import time

# Set the page layout and title
st.set_page_config(layout="wide", page_title="✨ Image Background Remover ✨")

# Instructions for the user
st.write("## 🖼️ Remove the background from your image!")
st.write(
    "Upload an image below, and watch the background magically disappear. 🎩✨"
    "\nOnce processed, you can download the final image or all images in a file. 💾"
)

# Maximum file size set to 20MB
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# Function to convert the image to bytes for download
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Function to handle the image processing
def fix_image(upload, bg_color="transparent"):
    image = Image.open(upload)  # Image.open expects a file-like object

    # Check if image is too large and resize if necessary
    if image.size[0] > 2000 or image.size[1] > 2000:
        st.warning("The image is quite large. It will be resized to fit optimally. 🔧")
        image = image.resize((2000, 2000))

    col1.write("### Original Image 📸")
    col1.image(image, use_container_width=True)

    # Show a loading spinner while the background is removed
    with st.spinner('🔄 Removing background...'):
        time.sleep(2)  # Simulate processing time
        fixed = remove(image)
        col2.write("### Processed Image 🛠️")

        # Add background color if selected
        if bg_color == "white":
            fixed = fixed.convert("RGBA")
            new_bg = Image.new("RGBA", fixed.size, (255, 255, 255, 255))
            new_bg.paste(fixed, (0, 0), fixed)
            fixed = new_bg
        elif bg_color == "black":
            fixed = fixed.convert("RGBA")
            new_bg = Image.new("RGBA", fixed.size, (0, 0, 0, 255))
            new_bg.paste(fixed, (0, 0), fixed)
            fixed = new_bg

        col2.image(fixed, use_container_width=True)

    return fixed

# Function to resize the image (resize before background removal)
def resize_image(image, width, height):
    return image.resize((width, height))

# Function to enhance image (Brightness, Contrast, etc.)
def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)

    return image

# Function to apply blur effect
def blur_image(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

# Function to zip images for multiple file download
def zip_images(images, file_names):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w') as zip_file:
        for img, name in zip(images, file_names):
            byte_img = convert_image(img)
            zip_file.writestr(name, byte_img)
    zip_buffer.seek(0)
    return zip_buffer

# Columns layout for displaying images
col1, col2 = st.columns(2)

# Main page image uploader and options
my_upload = st.file_uploader("Upload your image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed", accept_multiple_files=True)

# Image enhancement controls
brightness = st.slider("💡 Brightness", 0.5, 2.0, 1.0)
contrast = st.slider("🎨 Contrast", 0.5, 2.0, 1.0)
sharpness = st.slider("🔪 Sharpness", 0.5, 2.0, 1.0)

# Image effect controls
blur_radius = st.slider("🌫️ Blur Radius", 0, 10, 2)

if my_upload is not None:
    images = []
    file_names = []
    
    # Process multiple uploaded files
    for uploaded_file in my_upload:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"{uploaded_file.name} is too large. Please upload an image smaller than 20MB. 🚨")
            continue
        image = Image.open(uploaded_file)
        
        # Optionally resize image
        width = st.slider("📏 Width", 100, 2000, image.width)
        height = st.slider("📏 Height", 100, 2000, image.height)
        
        resized_image = resize_image(image, width, height)
        
        # Enhance image if needed
        enhanced_image = enhance_image(resized_image, brightness, contrast, sharpness)
        
        # Apply blur effect if selected
        final_image = blur_image(enhanced_image, blur_radius)
        
        # Select background color
        bg_color = st.selectbox("🎨 Choose Background Color", ["transparent", "white", "black"])
        
        # Convert final image to a BytesIO object before passing it to fix_image
        img_byte_arr = BytesIO()
        final_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        # Process image with the selected background color
        processed_image = fix_image(upload=img_byte_arr, bg_color=bg_color)
        
        images.append(processed_image)
        file_names.append(f"processed_{uploaded_file.name}")

    # Download processed images as a zip
    if images:
        zip_buffer = zip_images(images, file_names)
        st.sidebar.download_button("📥 Download All Processed Images", zip_buffer, "processed_images.zip", "application/zip")
else:
    st.write("🔔 Please upload an image to start processing.")

