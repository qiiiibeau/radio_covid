import streamlit as st
import os
from PIL import Image

from streamlit_utils import aligned_markdown

title = "Analyse de radiographies pulmonaires Covid-19"
sidebar_name = "Introduction"

image1_path = '../streamlit_app/assets/grad_cam_wide.jpg'
# image1 = Image.open(image1_path)


def run():
    # st.image(image1)

    st.title(title)

    st.markdown("---")

    st.title("Problématique")
    aligned_markdown(
        "Peut-on construire un modèle détectant efficacement les radiographies pulmonaires de patients positifs au "
        "Covid-19 ?")
    aligned_markdown(
        "Si la classification par deep learning permettait une détection efficace, elle pourrait être un outil de "
        "support dans les hôpitaux et cliniques en l’absence de test classique notamment lors de pics épidémiques."
    )

