import os
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import random
from skimage.io import imread, imshow

from streamlit_utils import aligned_markdown, load_resize_img, load_mask
from streamlit_utils import LST_GROUP, DIR_IMAGES, DIR_MASKS, NUM_ALL_IMG, get_fname



title = "Dataset"
sidebar_name = "Dataset"


image2_path = '../streamlit_app/assets/repartition_labels.png'
image2 = Image.open(image2_path)

def demonstration_random_img():
    image = plt.figure(figsize=(16, 13))
    for i in range(4):
        group = LST_GROUP[i]
        r_idx = random.choice(range(1, NUM_ALL_IMG[i] + 1))
        img = load_resize_img(group, r_idx)
        mask = load_mask(group, r_idx)
        img_mask = img * mask
        plt.subplot(3, 4, i + 1)
        imshow(img)
        plt.axis("off")
        plt.title(f"{group} - {r_idx}", fontsize=28)
        plt.subplot(3, 4, i + 5)
        imshow(mask)
        plt.axis("off")
        plt.subplot(3, 4, i + 9)
        imshow(img_mask)
        plt.axis("off")
        plt.tight_layout()
    return image


def run():
    st.title("Dataset")
    aligned_markdown(
        "Nous avons utilisé les données open source récoltées par une équipe de chercheurs de l’université du Qatar"
        " à Doha, l’université de Dhaka au Bangladesh avec leurs collaborateurs du Pakistan et de Malaisie ainsi que"
        " de médecins [Réf. 1]. Cette base contient 21 165 images de radiographies pulmonaires accompagnées de"
        " leurs masques respectifs et réparties dans quatre dossiers : « Covid » contenant 3 616 radiographies"
        " pulmonaires de patients positifs au covid-19, « Normal » contenant 10 192 radiographies de patients"
        " n’ayant pas de pathologie pulmonaire,  « Viral Pneumonia » contenant 1 345 radiographies de "
        "patients souffrant de pneumonie virale et « Lung Opacity » contenant 6 012 radiographies de patients "
        "ayant d’autres pathologies pulmonaires. Le format des images en 256 niveaux de gris est de 299 × 299 "
        "pixels. Le format des masques est en revanche de 256 × 256 pixels. \n"
    )
    st.markdown(
        """
        
        - 21 165 images (png, 299 × 299)
        - 21 165 masques (png, 256 × 256)
        - Une répartition très déséquilibrée
        
        

        """
    )
    st.image(image2)

    aligned_markdown("\n\nOn peut proposer de tirer au hasard une image par classe et afficher les images avec les masques"
                     " respectifs.\n")
    if st.button('tirer une image par classe'):
        image = demonstration_random_img()
        st.pyplot(image)
