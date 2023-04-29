import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from streamlit_utils import aligned_markdown

title = "Exploration des données"
sidebar_name = "exploration des données"

image1 = Image.open('../streamlit_app/assets/mean_img.jpg')
image2 = Image.open('../streamlit_app/assets/mean_maske.jpg')
image3 = Image.open('../streamlit_app/assets/std_img.jpg')
image4 = Image.open('../streamlit_app/assets/std_maske.jpg')



def run():
    st.title(title)

    st.markdown(
        """
        ## Des données augmentées
        """
    )

    aligned_markdown(
        """
        Nous avons mené une première analyse qualitative sur un échantillon de 10 % des images (2 116) suivant 
        les quatre classes et avons identifié cinq familles de caractéristiques:  transformation de l'image : 
        certaines images semblent avoir été transformées probablement par un générateur même si cela n’est pas précisé
         dans la documentation."""
    )

    st.markdown("- cadrage : la zone pulmonaire n’est pas toujours complète sur l’image.")
    st.button("afficher une exemple de cadrage")
    st.markdown("- luminosité : la luminosité peut varier de façon importante.")
    st.button("afficher une exemple de luminosité")
    st.markdown("- bruit ou flou : quelques images sont particulièrement floues ou avec du bruit. ")
    st.button("afficher une exemple de bruit ou flou")
    st.markdown("- interférences : des objets (électrodes, tubes, équipements médicaux, annotations) apparaissent"
                " parfois dans le cadre ou dans la zone pulmonaire.")
    st.button("afficher une exemple d'interférences")

    st.image(Image.open("../streamlit_app/assets/anormalie.png"))

    st.markdown(
        """
        ##  Images moyennes 

        """
    )

    aligned_markdown(
        """
        Pour prendre en compte la structure spatiale des images et avoir une idée de la forme moyenne par classe, nous 
        avons effectué la moyenne des images avec et sans masque. La figure 1.8 montre que certains éléments en dehors 
        de la zone pulmonaire pourraient être considérés comme caractéristiques d’une classe (e.g. la taille d’une zone
         noire autour du thorax ou la position des bras).
        """
    )

    st.image(image1)
    st.image(image2)

    aligned_markdown(
        """
        La figure montre que la forme ou l’écartement entre les poumons pourraient être considérés comme 
        caractéristiques.
        """
    )

    st.markdown(
        """
        ## Images écart type

        """
    )

    aligned_markdown(
        """
        Pour avoir une idée de la façon dont chaque image varie au sein de chaque classe, nous avons calculé l’écart 
        type des images. Les écarts type confirment une variation des images autour de formes qui pourraient être 
        considérées comme caractéristiques de chaque classe."""
    )

    st.image(image3)
    st.image(image4)

    st.markdown(
        """
        \n
        \n
        ## Isomap
        
        """
    )
    st.image(Image.open('../streamlit_app/assets/isomap_avec_mask.jpg'))
    aligned_markdown(
        """
        Les images avec masque rangées sur la gauche sont plus sombres, présentent un espace plus important entre les 
        poumons et des côtes plus visibles. Les images rangées sur la droite sont plus claires et les côtes sont moins
         visibles. En haut, la forme des poumons semble plus allongée, tandis que dans la partie inférieure du 
         graphique, la forme des poumons semble plus compacte. Cela pourrait représenter des biais dans la 
         classification.
        """
    )
    st.image(Image.open('../streamlit_app/assets/isomap_sans_mask.jpg'))
    aligned_markdown(
        """
        Dans le cas des images non masquées, nous observons une organisation différente, mais la clarté moyenne, la 
        visibilité des côtes et la forme semblent rester des critères de rangement.
        """
    )