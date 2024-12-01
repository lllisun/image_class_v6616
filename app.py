import streamlit as st
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the VGG16 model, pretrained on ImageNet
model = VGG16(weights='imagenet')

st.title("Image Classification with VGG16")
st.header("Upload an image for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image for model
    img = img.resize((224, 224))  # Resize image to match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_array)

    # Predict
    if st.button('Predict'):
        predictions = model.predict(img_processed)
        # Get top 3 predictions
        top_preds = decode_predictions(predictions, top=3)[0]
        
        st.write('Predictions (top 3):')
        for pred in top_preds:
            st.write(f"{pred[1]}: {pred[2]*100:.2f}% confidence")
