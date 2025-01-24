import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from models.acne.acne_model import MyNet  # Replace with your actual model class
from recommender.recommend import Recommender


# Paths and features
products_path = 'recommender/essential_skin_care.csv'
features = [
    'normal', 'dry', 'oily', 'combination', 'sensitive', 'general care',
    'hydration', 'dull skin', 'dryness', 'softening', 'smoothening',
    'fine lines', 'wrinkles', 'acne', 'blemishes', 'pore care',
    'daily use', 'dark circles'
]

# Load Acne Model - Hide errors
try:
    acne_model = MyNet()  # Replace with your actual model class
    acne_model.load_state_dict(torch.load('saved_models/acne-severity/best_6.pt', map_location='cpu'))
    acne_model.eval()
except Exception:
    acne_model = None  # Set the model to None if loading fails

# Load Skin Type Model - Hide errors
try:
    skintype_model = MyNet()  # Replace with your skin type model class
    skintype_model.load_state_dict(torch.load('saved_models/skintype/best_20.pt', map_location='cpu'))
    skintype_model.eval()
except Exception:
    skintype_model = None  # Set the model to None if loading fails

# Load Recommender
recommender = Recommender(products_path, features)


def predict_acne_level(image):
    if acne_model is None:
        return None  # Return None if model is not loaded
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = acne_model(input_tensor)
        probs = nn.Softmax(dim=1)(output)
        predicted_class = torch.argmax(probs, axis=1).item()
    return predicted_class


def predict_skin_type(img):
    if skintype_model is None:
        return None  # Return None if model is not loaded

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = skintype_model(input_tensor)
        predicted_class = torch.argmax(output, axis=1).item()
    return predicted_class


def get_feature_vector(skin_type, acne_level, selected_features):
    fv = [0] * len(features)
    if skin_type == 'Normal':
        fv[0] = 1
    elif skin_type == 'Dry':
        fv[1] = 1
    else:
        fv[2] = 1
    if acne_level >= 2:  # Severe or Extreme
        fv[13] = 1
    for feature in selected_features:
        fv[features.index(feature)] = 1
    return fv


def get_essential_skincare(fv):
    return recommender.recommend(fv)


def main():
    st.title("Skin Care Essentials")
    st.header("Personalized Skin Care Recommendations")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict acne severity and skin type
        if acne_model is not None:
            acne_level = predict_acne_level(image)
            severity_levels = ['Clear', 'Mild', 'Severe', 'Extreme']
            severity = severity_levels[acne_level] if acne_level is not None else "Unable to Predict"
            st.subheader("Acne Severity Level")
            st.write(severity)
        else:
            st.subheader("Acne Severity Level")
            st.write("Model not loaded")

        if skintype_model is not None:
            skin_type_idx = predict_skin_type(image)
            skin_types = ['Normal', 'Dry', 'Oily']
            skin_type = skin_types[skin_type_idx] if skin_type_idx is not None else "Unable to Predict"
            st.subheader("Skin Type")
            st.write(skin_type)
        else:
            st.subheader("Skin Type")
            st.write("Model not loaded")

        selected_features = st.multiselect("Select additional concerns", features)

        # Generate recommendations
        if st.button("Get Recommendations"):
            if acne_model is not None and skintype_model is not None:
                feature_vector = get_feature_vector(skin_type, acne_level, selected_features)
                recommendations = get_essential_skincare(feature_vector)

                for category, products in recommendations.items():
                    st.subheader(category.capitalize())
                    for product in products:
                        st.markdown(f"""
                        **{product['name']}**  
                        *{product['brand']}*  
                        **Price:** {product['price']}  
                        **Skin Type:** {product['skin type']}  
                        **Concerns:** {', '.join(product['concern'])}  
                        [Buy Now]({product['url']})
                        """)
            else:
                st.write("Models not available for prediction")


if __name__ == '__main__':
    main()
