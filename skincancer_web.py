pip install tensorflow
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st
import io

# Function to preprocess images
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize the image
    return img

# Function to create the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Streamlit interface
st.title("Skin Cancer Detection")

# Upload dataset
uploaded_files = st.file_uploader("Upload your dataset images (JPG)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    images = []
    labels = []
    
    # Process uploaded images
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append(preprocess_image(image))
        # Assuming the filename contains 'malignant' for positive cases
        labels.append(1 if "malignant" in uploaded_file.name else 0)

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    st.success("Model trained successfully!")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image for prediction (JPG)", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)

    if prediction[0][0] > 0.5:
        st.write("Prediction: Malignant")
    else:
        st.write("Prediction: Benign")
