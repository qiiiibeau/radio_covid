import os.path

import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
from skimage.io import imshow
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from streamlit_utils import aligned_markdown, load_resize_img, load_mask, NUM_ALL_IMG

title = "Benchmark des modèles"
sidebar_name = "modèle benchmarking"

DIR_MODEL = '../result/models/'


def run():
    st.title(title)
    st.markdown(
        """
        ## Baseline Models
        """
    )