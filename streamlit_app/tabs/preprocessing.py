import streamlit as st
from PIL import Image


title = "Preprocessing pipeline"
sidebar_name = "preprocessing"


def run():

    st.title(title)

    st.markdown(
        """
        ## Preprocessing sans masque
        """
    )

    st.image(Image.open('../streamlit_app/assets/pipeline_sans_mask.jpg'))


    st.markdown(
        """
        ## Preprocessing avec masque
        """
    )

    st.image(Image.open('../streamlit_app/assets/pipeline_avec_mask.jpg'))