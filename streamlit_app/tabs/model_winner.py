import os

import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
from skimage.io import imshow
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

from streamlit_utils import aligned_markdown, load_resize_img, load_mask, NUM_ALL_IMG

title = "Le modèle retenu"
sidebar_name = "Le model"

DIR_MODEL = '../result/models/'

def load_vgg_predict_label(model_version, image, mask):
    accuracy = 0.11111
    return 0, accuracy

def model_prediction(model, image_3):
  prediction = model.predict(image_3)
  return np.argmax(prediction[0])

# input X_1 is a serie of images with shape : (len(X),256,256,1)
# for just one image the shape is : (1,256,256,1)
# X_1 = X.reshape(len(X),256,256,1)


def unet_prediction(X_1):
    model = load_model("path/unet_256_5200_20ep_batch_32.h5") #update path
    masks = model.predict(X_1)
    masks = tf.argmax(masks, axis=-1)
    masks = masks[..., tf.newaxis]
    return tf.cast(masks, tf.float32).numpy() #return the musk as an array



def run():
    st.title(title)
    st.markdown(
        """
        ## Le modèle de VGG16 unfreezed
        """
    )

    aligned_markdown("add description")

    st.image(Image.open("../streamlit_app/assets/vgg16_unfreezed.jpg"))
    st.markdown(
        """
        ## Evaluation du modèle
        """
    )
    st.image(Image.open("../streamlit_app/assets/metric_vgg16_unfreezed.png"))
    aligned_markdown("add description")

    ## todo insert dataframe matrice de confusion
    st.text("todo insert dataframe ou schéma matrice de confusion")

    st.markdown(
        """
        ## Test 
        """
    )



    option = st.selectbox(
        "Chosir label",
        ('Covid', 'Normal', 'Viral Pneumonia', 'Lung Opacity'))

    option_label_mapper = {
        "Covid": "covid",
        "Normal": "normal",
        'Viral Pneumonia': "viral",
        'Lung Opacity': "opac"
    }

    label_id_mapper = {
        "covid": 0,
        "normal": 1,
        "viral": 2,
        "opac": 3
    }

    label = option_label_mapper[option]

    image_source = st.radio(
        "Image source",
        ('aléatoirement de Kaggle dataset', 'télécharger votre radiographie')
    )

    mask_source = None
    idx = None
    if image_source == 'télécharger votre radiographie':
        uploaded_file = st.file_uploader("Télécharge l'image")
        if uploaded_file is not None:
            # To read file as bytes:
            img = uploaded_file.getvalue()
            img_title = "votre image radio téléchargé"

            fig = plt.figure(figsize=(5, 5))
            imshow(img)
            plt.title(img_title)
            plt.axis('off')
            st.pyplot(fig)

            mask_source = st.radio(
                "Mask source",
                ('télécharger votre fichier de masque', 'généré par U-Net')
            )

    else:
        idx = random.choice(range(1, NUM_ALL_IMG[label_id_mapper[label]] + 1))
        img = load_resize_img(label, idx)
        img_title = label + '-' + str(idx)

        fig = plt.figure(figsize=(5, 5))
        imshow(img)
        plt.title(img_title)
        plt.axis('off')
        st.pyplot(fig)

        mask_source = st.radio(
            "Mask source",
            ('De Kaggle dataset', 'généré par U-Net')
        )

    if mask_source == 'De Kaggle dataset':
        mask = load_mask(label, idx)
    elif mask_source == 'télécharger votre fichier de masque':
        uploaded_masque = st.file_uploader("Télécharge le masque")
        if uploaded_masque is not None:
            # To read file as bytes:
            mask = uploaded_masque.getvalue()
    elif mask_source == "généré par U-Net":
        model = load_model(os.path.join(DIR_MODEL, "unet_256_5200_20ep_batch_32.h5"))
        mask = model.predict(img)

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    imshow(mask)
    plt.title("masque")
    plt.axis('off')
    plt.subplot(122)
    imshow(mask * img)
    plt.title("image masqué")
    plt.axis('off')

    st.pyplot(fig)

    # todo first select model then select and show mask (incase reverse)
    lst_models = ['VGG16 unfreezed entrainé sur images sans masque',
         'VGG16 unfreezed entrainé sur images avec masque',
         'VGG16 unfreezed entrainé sur images avec masque inversé']

    model_version_name = st.selectbox(
        "Chosir le modèle entrainé",
        (lst_models))

    model_version = {zip(lst_models, range(3))}[model_version_name]

    if st.button("load le modèle entrainé et obtenir la prédiction"):

        st.text("todo insert the function de Patrizia")
        y_pred, accuracy = load_vgg_predict_label(model_version, img, mask)
        print("m")

    # todo grad cam



