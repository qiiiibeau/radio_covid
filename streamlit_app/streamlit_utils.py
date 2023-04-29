import os
import sys

import streamlit as st
from skimage.io import imread
from skimage.transform import resize


sys.path.append('../code')
# print(sys.path)
from utils import LST_GROUP, LST_FOLDERS, NUM_ALL_IMG, get_fname

DIR_DATA_LOCAL = '../data/COVID-19_Radiography_Dataset/'
DIRS = dict(zip(LST_GROUP, [os.path.join(DIR_DATA_LOCAL, folder) for folder in LST_FOLDERS]))
DIR_IMAGES = dict(zip(LST_GROUP, [os.path.join(dir, 'images') for dir in DIRS.values()]))
DIR_MASKS = dict(zip(LST_GROUP, [os.path.join(dir, 'masks') for dir in DIRS.values()]))


def aligned_markdown(text):
    st.markdown(f'<div style="text-align: justify;">{text}</div>', unsafe_allow_html=True)


def get_path_img(group, idx):
    return os.path.join(DIR_IMAGES[group], get_fname(group, idx))


def get_path_mask(group, idx):
    return os.path.join(DIR_MASKS[group], get_fname(group, idx))


def load_resize_img(group, idx):
    path_img = get_path_img(group, idx)
    return resize(imread(path_img, as_gray=True), (256, 256), anti_aliasing=True)


def load_mask(group, idx):
    path_mask = get_path_mask(group, idx)
    return imread(path_mask, as_gray=True)