import base64
import csv
from datetime import datetime
import folium
from folium.plugins import HeatMap
from functools import partial
import geocoder
#from geopy.geocoders import Nominatim
import io
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageFont
import shutil
import streamlit as st
from streamlit_folium import folium_static
import tensorflow as tf
from urllib.request import urlopen
from urllib.parse import urlparse

#geolocator = Nominatim(user_agent="geoapiExercises")

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
    text = '<p style="color:Black; font-size: 12px"> 2) Folder Upload </p>'
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
                            ['Image Upload', 'Image URL', 'Folder Upload'])
if page == 'Image Upload':
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose Image...", type=["jpg", "jpeg"])
elif page == 'Image URL':
    with st.sidebar:
        url = st.text_input("Enter Image URL:")
else:
    with st.sidebar:
        folder = st.text_input('Enter name of folder:')
if page == 'Folder Upload':
    output = st.sidebar.text_input('Enter output folder:', value = 'output')
weights = ['Weights/Timm_ResNet152V2.h5',
           'Weights/El_Hassan_EfficientNetB0.hdf5',
           'Weights/Timm_MobileNetV2.h5',
           'Weights/El_Hassan_EfficientNetB0_Rut.hdf5']
with st.sidebar: 
    if page == 'Folder Upload':
        csv_upload = st.file_uploader('Choose location file', type=['csv'])
completed = False # Flag to stop prediction from running indefinitely
with st.sidebar:
    btn = st.button('Predict')
save_label = ""
classes = ['Crack', 'Groove', 'Rut', 'Subsidience']

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

def predict_all(data):
    prob_dict = {}
    for weight in weights:
        model = tf.keras.models.load_model(weight)
        try:
            probability = model.predict(data)
        except:
            print('Cannot run.')
        prediction_probability = probability[0, probability.argmax(axis=1)][0]
        label = classes[int(np.argmax(probability, axis=1))]
        prob_dict[label] = prediction_probability
    max_key = max(prob_dict, key=prob_dict.get)
    max_val = prob_dict[max_key]
    if max_val > 1:
        max_val /= 2
    return (max_key, max_val)

# TODO: Add a parameter to save results to "output" and output that to ZIP
def predict_image(img, image_size, outpt, save, filename):
    with col2:
        print("IMAGE_FILENAME", filename)
        filename = filename.split('/')[-1]
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
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
        label, prediction_probability = predict_all(data)
        text = '<p style="color:Black; font-size: 18px; font-weight: Bold"> Defect Type and Accuracy: </p>'
        st.markdown(text, unsafe_allow_html=True)
        st.write("""
        Predicted: {}\n
        Confidence: {:.5f}
        """.format(label, prediction_probability * 100))
        if not os.path.exists(outpt):
            os.makedirs(outpt)
        img1 = ImageDraw.Draw(img)
        #img1.rectangle([230, 200, 250, 250], fill ="#ffff33", outline ="red")
        fnt = ImageFont.truetype("fonts/arial.ttf", 20)
        img1.text((10, 10), label + "\n" + str(round(prediction_probability * 100000) / 100000), font=fnt, fill="red", align="left")
        print(f'{outpt}/{filename}')
        img.save(f'{outpt}/{filename}', 'jpeg')
    if not save: 
        g = geocoder.ip('me')
        print(g.latlng)
        m = folium.Map(location=g.latlng)
        img_contents = io.BytesIO()
        img.save(img_contents, format="jpeg")
        img_contents = img_contents.getvalue()
        encoded = base64.b64encode(img_contents)
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = folium.IFrame(html(encoded.decode('UTF-8')), width=image_size+20, height=image_size+20)
        popup = folium.Popup(iframe, max_width=400)
        folium.Marker(
            g.latlng, popup=popup, tooltip=label
        ).add_to(m)
        st_map = folium_static(m, width=700, height=450)
    else:
        global save_label
        save_label = label

def predict_multiple(folder, image_size):
    if os.path.isdir(output):
        shutil.rmtree(output)
    g = geocoder.ip('me')
    print(g.latlng)
    m_r = folium.Map(location=g.latlng)
    m_s = folium.Map(location=g.latlng)
    m_c = folium.Map(location=g.latlng)
    m_g = folium.Map(location=g.latlng)
    m_t = folium.Map(location=g.latlng)
    if csv_upload is not None:
        df = pd.read_csv(csv_upload)
        locations = df.values.tolist()
    else: 
        with open('locations/locations.csv', newline='') as f:
            reader = csv.reader(f)
            locations = list(reader)
    i = 0
    for filename in os.scandir(folder):
        if filename.is_file():
            img = Image.open(filename.path)
            predict_image(img, image_size, output, True, filename.path)
            img_contents = io.BytesIO()
            img.save(img_contents, format="jpeg")
            img_contents = img_contents.getvalue()
            encoded = base64.b64encode(img_contents)
            html = '<img src="data:image/png;base64,{}">'.format
            iframe = folium.IFrame(html(encoded.decode('UTF-8')), width=image_size+20, height=image_size+20)
            popup = folium.Popup(iframe, max_width=400)
            floats = [float(x) for x in locations[i]]
            print("LABEL", save_label)
            if save_label == 'Rut':
                HeatMap([floats]).add_to(m_r)
            elif save_label == 'Crack':
                HeatMap([floats]).add_to(m_c)
            elif save_label == 'Groove':
                print(i)
                HeatMap([floats]).add_to(m_g)
            else:
                HeatMap([floats]).add_to(m_s)
            folium.Marker(
                locations[i], popup=popup, tooltip=save_label
            ).add_to(m_t)
            i += 1
    st_map_r = folium_static(m_r, width=700, height=450)
    st_map_c = folium_static(m_c, width=700, height=450)
    st_map_g = folium_static(m_g, width=700, height=450)
    st_map_s = folium_static(m_s, width=700, height=450)
    st_map_t = folium_static(m_t, width=700, height=450)

if btn and not completed:
    completed = True
    if page == 'Image Upload':
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            predict_image(img, 256, 'output_folder', False, uploaded_file.name)
    elif page == 'Image URL':
        if url:
            img = Image.open(urlopen(url))
            a = urlparse(url)
            predict_image(img, 256, 'output_folder', False, os.path.basename(a.path))
    else:
        if folder and os.path.exists(folder) and os.listdir(folder):
            predict_multiple(folder, 256)

    completed = False