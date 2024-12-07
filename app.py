import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Define a function for preprocessing the image
def preprocess_image(image, image_size):
    # Convert image to OpenCV format
    img = np.array(image)
    # Resize the image to match the input size of the model
    img = cv2.resize(img, (image_size, image_size))
    # Perform any additional preprocessing steps if necessary (e.g., normalization)
    # Return the preprocessed image
    return img

# Define a function for predicting the class of an image
def predict_image(model, image, class_names):
    # Preprocess the image
    preprocessed_img = preprocess_image(image, model.input_shape[1])
    # Expand the dimensions to match the model's input shape
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    # Make predictions using the model
    predictions = model.predict(preprocessed_img)
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

# Load the trained model
model_path = 'myModel.h5'
model = load_model(model_path)

# Define class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

st.markdown("### Welcome to our mini project")
st.title("IS IT A BRAIN TUMOR OR HEALTHY BRAIN!")
st.header("Detection of MRI Brain Tumor")
st.text("Upload a brain MRI Image to detect tumor")

uploaded_pic = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"]) 

if uploaded_pic is not None:
    test_image = Image.open(uploaded_pic)
else:
    test_image = Image.open("Example.jpg")
    st.write("This is an example image:")
    st.image(test_image, caption='Example Image', width=200)

# Resize the image for display
resized_image = test_image.resize((150, 150))
st.image(resized_image, caption='Uploaded Image.', use_column_width=True)

if st.button("SUBMIT"):
    st.markdown("#### CLASSIFYING......")
    predicted_class = predict_image(model, test_image, class_names)
    st.write(f'Predicted class: {predicted_class}')

if st.button("Contact Us"):
    st.text("Email us at 21051731@kiit.ac.in, 21051752@kiit.ac.in")

# GitHub link
st.markdown('[GitHub Repository](https://github.com/your-username/your-project)', unsafe_allow_html=True)

with st.expander("Click here to read more"):
    st.write("This project is a collaboration of Ayush Sarkar and Pratyush Kumar Shrivastava")
