import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
from skimage.io import imshow
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from streamlit_utils import aligned_markdown, load_resize_img, load_mask, NUM_ALL_IMG, LST_GROUP

title = "Lung mask segmentation by U-net"
sidebar_name = "mask segmentation"


def get_mask_unet(image):
    # todo
    return image

def run():
    st.title(title)

    st.markdown("")

    st.image(Image.open('../streamlit_app/assets/u_net_architecture.png'))

    aligned_markdown("U-Net entraîné sur 5 200 images au format  256 × 256 et a atteint une accuracie de "
                     "99%")

    aligned_markdown("Vous pouvez charger un image aléatoire de Kaggle dataset, afficher le masque correspondant "
                     "fourni et le masque prédit par notre modèle u-net")

    if st.button("test un image aléatoire"):
        label_id = random.choice(range(1, 5))
        label = LST_GROUP[label_id]
        idx = random.choice(range(1, NUM_ALL_IMG[label_id] + 1))
        img_title = label + '-' + str(idx)
        # load image Kaggle
        img = load_resize_img(label, idx)
        # load mask Kaggle
        mask = load_mask(label, idx)
        # create mask by unet model
        mask_unet = get_mask_unet(img)


        fig = plt.figure(figsize=(15, 5))
        plt.subplot(131)
        imshow(img)
        plt.title(img_title)
        plt.axis('off')

        plt.subplot(132)
        imshow(mask)
        plt.axis('off')

        plt.subplot(133)
        imshow(mask_unet)
        plt.axis('off')
        st.pyplot(fig)

