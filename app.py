import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

model = load_model('Signature_Detection_Model.h5') #add final model file
class_dict = np.load("signature_Forgery_Detection.npy") #labels file we have saved as array

def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        base64_img = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_img}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
contnt = "<p>A handwritten signature verification App trained on the Handwritten Signature Datasets in Kaggle with ResNet50.</p> " 
         

if __name__ == '__main__':
    add_bg_from_local("Background.jpg")
    new_title = '<p style="font-family:sans-serif; color:red; font-size: 50px;">Handwritten Signature Verification App</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    contnt = '<p style="font-family:sans-serif; color:white; font-size: 20px;">A signature is very important for presenting a person's acceptance. So there are many chances to make it forged. To avoid this introducing a free trail,</p>'
    st.markdown(contnt,unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((300, 300))
        st.image(img)
        if st.button("Predict"):
            pred = predict(img)
            name = class_dict[pred]
            result = f'<p style="font-family:sans-serif; color:Red; font-size: 32px;">The given image is {name}</p>'
            st.markdown(result, unsafe_allow_html=True)
