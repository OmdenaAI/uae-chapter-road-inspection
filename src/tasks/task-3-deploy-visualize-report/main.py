import numpy as np
import streamlit as st
import tensorflow as tf
from urllib.request import urlopen
from PIL import Image, ImageOps
from datetime import datetime
from streamlit_js_eval import get_geolocation
from functools import partial
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://pisces.bbystatic.com/image2/BestBuy_US/images/products/6520/6520004cv12d.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


add_bg_from_url()

# Space out the maps so the first one is 3x the size of the other three
col1, col2 = st.columns((3, 1))
with col1:
    img = Image.open('omdena uae.jpg')
    st.image(img, width=150)
    # st.title("Machine Learning Approach to Detect Road Defects: A Contribution toward Road Safety")
    text = '<p style="color:Black; font-size: 24px; font-weight: Bold"> Machine Learning Approach to Detect Road Defects: A Contribution toward Road Safety </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 18px; font-weight: Bold"> Brief: </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> Road surfaces degrade daily as a result of heavy traffic and rain. Such degradation will impact the driver’s comfort and economic efficiency. Municipalities perform regular inspections to maintain roads and ensure driving safety. However, current practices of performing road inspections are time-consuming and labor-intensive. The project aims to use machine learning to automatically detect road defects: cracks, grooves, ruts, and subsidence.  </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 18px; font-weight: Bold"> Instructions to use: </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 14px; font-weight: Bold"> Choose your option for image input </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> 1) Image Upload </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> 2) Image URL </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 14px; font-weight: Bold"> Choose your model </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> 1) ResNet152V2 (Crack & Groove) </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> 2) Deep Hybrid EfficientNetB0 (Crack & Groove) </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> 3) MobileNetV2 (Rut & Subsidence) </p>'
    st.markdown(text, unsafe_allow_html=True)
    text = '<p style="color:Black; font-size: 12px"> 4) Deep Hybrid EfficientNetB0 (Rut & Subsidence) </p>'
    st.markdown(text, unsafe_allow_html=True)

page = st.sidebar.selectbox('UPLOAD IMAGE OR IMAGE URL',
                            ['Image Upload', 'Image URL'])
if page == 'Image Upload':
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose Image...", type=["jpg", "jpeg"])
else:
    with st.sidebar:
        url = st.text_input("Enter Image URL:")
model = st.sidebar.selectbox('CHOOSE THE MODEL',
                             ['ResNet152V2 (Crack & Groove)',
                              'Deep Hybrid EfficientNetB0 (Crack & Groove)',
                              'MobileNetV2 (Rut & Subsidence)',
                              'Deep Hybrid EfficientNetB0 (Rut & Subsidence)'])


def get_location():
    try:
        loc = get_geolocation()
        Latitude = loc['coords']['latitude']
        Longitude = loc['coords']['longitude']
        time = datetime.fromtimestamp(loc['timestamp'] // 1000)

        reverse = partial(geolocator.reverse, language="en")
        location = reverse(str(Latitude) + "," + str(Longitude))
        address = location.raw['address']

        # traverse the data
        city = address.get('city', '')
        state = address.get('state', '')
        country = address.get('country', '')
        st.write("""
        Location: {} {}, {}.\n
        Time : {}\n
        """.format(city, state, country, time))
    except:
        st.write("Getting GeoLocation...")


def predict_with_image(image, weights, image_size, classes):
    img = Image.open(image)
    with col2:
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        model = tf.keras.models.load_model(weights)
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, image_size, image_size, 3), dtype=np.float32)
        # image sizing
        size = (image_size, image_size)
        img = ImageOps.fit(img, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(img)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        probabilities = model.predict(data)
        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
        prediction = model.predict(data)
        label = classes[int(np.argmax(probabilities, axis=1))]
        text = '<p style="color:Black; font-size: 18px; font-weight: Bold"> Defect Type and Accuracy: </p>'
        st.markdown(text, unsafe_allow_html=True)
        st.write("""
        Predicted: {}\n
        Confidence: {:.5f}
        """.format(label, prediction_probability * 100))
        get_location()


def predict_with_url(url, weights, img_size, classes):
    img = Image.open(urlopen(url))
    with col2:
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        model = tf.keras.models.load_model(weights)
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, img_size, img_size, 3), dtype=np.float32)
        # image sizing
        size = (img_size, img_size)
        image = ImageOps.fit(img, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        probabilities = model.predict(data)
        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
        prediction = model.predict(data)
        label = classes[int(np.argmax(probabilities, axis=1))]
        text = '<p style="color:Black; font-size: 18px; font-weight: Bold"> Defect Type and Accuracy: </p>'
        st.markdown(text, unsafe_allow_html=True)
        st.write("""
        Predicted: {}\n
        Confidence: {:.5f}
        """.format(label, prediction_probability * 100))
        get_location()

if model == 'ResNet152V2 (Crack & Groove)':
    classes = ["Crack", "Groove"]
    if page == 'Image Upload':
        if uploaded_file is not None:
            predict_with_image(uploaded_file, 'Weights/Timm_ResNet152V2.h5', 256, classes)
    else:
        if url:
            predict_with_url(url, 'Weights/Timm_ResNet152V2.h5', 256, classes)
elif model == 'Deep Hybrid EfficientNetB0 (Crack & Groove)':
    classes = ["Crack", "Groove"]
    if page == 'Image Upload':
        if uploaded_file is not None:
            predict_with_image(uploaded_file, 'Weights/El_Hassan_EfficientNetB0.hdf5', 224, classes)
    else:
        if url:
            predict_with_url(url, 'Weights/El_Hassan_EfficientNetB0.hdf5', 224, classes)
elif model == 'MobileNetV2 (Rut & Subsidence)':
    classes = ["Rut", "Subsidence"]
    if page == 'Image Upload':
        if uploaded_file is not None:
            predict_with_image(uploaded_file, 'Weights/Timm_MobileNetV2.h5', 256, classes)
    else:
        if url:
            predict_with_url(url, 'Weights/Timm_MobileNetV2.h5', 256, classes)
else:
    classes = ["Rut", "Subsidence"]
    if page == 'Image Upload':
        if uploaded_file is not None:
            predict_with_image(uploaded_file, 'Weights/El_Hassan_EfficientNetB0_Rut.hdf5', 224, classes)
    else:
        if url:
            predict_with_url(url, 'Weights/El_Hassan_EfficientNetB0_Rut.hdf5', 224, classes)

