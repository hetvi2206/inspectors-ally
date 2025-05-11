import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import get_train_test_loaders
from utils.model import CustomVGG

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

# App Title and Description
st.title("InspectorsAlly")
st.caption("Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App")
st.write("Upload a product image or use your camera — our AI will classify it as Good or Anomaly.")

# Sidebar content
with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )

    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Bottle Product Images."
    )

# Image loader function
def load_uploaded_image(file):
    return Image.open(file)

# Image Input Option
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

uploaded_file_img, camera_file_img = None, None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset folder
data_folder = os.path.join("./data/", "bottle")

# Anomaly Detection function
"""
Given an image path and a trained PyTorch model, returns the predicted class and bounding boxes for any defects detected in the image.
"""

def Anomaly_Detection(image, root):
    batch_size = 1

    model_path = "./weights/bottle_model.pth"
    model = CustomVGG()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _, test_loader = get_train_test_loaders(root, batch_size=batch_size)
    class_names = test_loader.dataset.classes

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs, _ = model(image)  # unpack probs and heatmap

    predicted_probabilities = probs.squeeze().cpu().numpy()

    # 🔍 Debug: print predictions
    print("Predicted Probabilities:", predicted_probabilities)
    print("Class Names:", class_names)
    
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = class_names[predicted_class_index]

    if predicted_class == "Good":
        return "✅ Congratulations! Your product has been classified as a 'Good' item with no anomalies detected."
    else:
        return "⚠️ We're sorry to inform you that our AI-based visual inspection system has detected an anomaly."

# Prediction Trigger
if st.button(label="Submit a Bottle Product Image"):
    st.subheader("Output")
    if input_method == "File Uploader" and uploaded_file_img:
        prediction = Anomaly_Detection(uploaded_file_img, data_folder)
    elif input_method == "Camera Input" and camera_file_img:
        prediction = Anomaly_Detection(camera_file_img, data_folder)
    else:
        st.warning("Please upload or capture an image first.")
        prediction = None

    if prediction:
        with st.spinner(text="This may take a moment..."):
            st.write(prediction)
