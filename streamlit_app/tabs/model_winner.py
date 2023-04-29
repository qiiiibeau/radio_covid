import os

import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
from skimage.io import imshow
import matplotlib.pyplot as plt
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.models import load_model
# tf.compat.v1.disable_eager_execution() # needed for Grad-CAM
from keras import backend as K
import cv2


from streamlit_utils import aligned_markdown, load_resize_img, load_mask, NUM_ALL_IMG, LST_GROUP, LABEL_MAPPER


ID2LABEL_MAPPER = dict(zip(range(4), LST_GROUP))
OPT2LABEL_MAPPER = {
    "Covid": "covid",
    "Normal": "normal",
    'Viral Pneumonia': "viral",
    'Lung Opacity': "opac"
}

title = "Le modèle retenu"
sidebar_name = "Le model"

DIR_MODEL = '../result/models/'


def unet_prediction(X_1):
    model = load_model(os.path.join(DIR_MODEL, "unet_256_5200_20ep_batch_32.h5"), compile=False)
    model.compile(loss="sparse_categorical_crossentropy")
    masks = model.predict(X_1)
    masks = tf.argmax(masks, axis=-1)
    masks = masks[..., tf.newaxis]
    return tf.cast(masks, tf.float32).numpy()
    # return tf.cast(masks, tf.float32).eval()


def unet_pipeline(img, inverse=False):
    """
    input X_1 is a serie of images with shape : (len(X),256,256,1)
    for just one image the shape is : (1,256,256,1)
    X_1 = X.reshape(len(X),256,256,1)
    """
    X_1 = img.reshape(1, 256, 256, 1)
    mask = unet_prediction(X_1)
    X = X_1 * mask
    X = X.reshape(256, 256)
    mask = mask.reshape(256, 256)
    return mask, X




def vgg_preprocessing(X):
    X = resize(X, (200, 200), anti_aliasing=True)
    X_vgg = np.zeros((200, 200, 3))
    # for i in range(3):
    #     for k in range(200):
    #         for h in range(200):
    #             X_vgg[k, h, i] = X_train[1, k, h]
    for i in range(3):
        X_vgg[:, :, i] = X
    X_vgg = X_vgg.reshape(1, X_vgg.shape[0], X_vgg.shape[1], X_vgg.shape[2])
    return X_vgg


def vgg_prediction(model_fname, X, grad_cam=True):
    model = load_model(os.path.join(DIR_MODEL, model_fname), compile=False)
    model.compile(loss = "categorical_crossentropy")
    X = vgg_preprocessing(X)
    prediction = model.predict(X)
    target_class = np.argmax(prediction[0])
    accuracy = 0  # todo
    predicted_label = ID2LABEL_MAPPER[target_class]

    res_grad_cam = None
    if grad_cam:
        class_output = model.output[:, target_class]
        last_conv_layer = model.get_layer('block5_conv3')
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([X])
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = X.reshape((X.shape[1], X.shape[2], X.shape[3]))*255
        heatmap_2 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_2 = np.uint8(255*heatmap_2)
        heatmap_2 = cv2.applyColorMap(heatmap_2, cv2.COLORMAP_JET)
        res_grad_cam = heatmap_2 * 0.4 + img  # superposed_image (heatmap+image)
    return predicted_label, res_grad_cam


# todo first select model then select and show mask (incase reverse)
lst_models = [
    'VGG16 unfreezed entrainé sur images avec masque',
    'VGG16 unfreezed entrainé sur images avec masque unet',
    'VGG16 unfreezed entrainé sur images sans masque',
    'VGG16 unfreezed entrainé sur images avec masque inversé'
]

lst_model_fname = [
    'uvgg16_f_200_balanced_masked.h5',
    'uvgg16_f_200_balanced_unet.h5',
    'uvgg16_f_200_balanced_unmasked.h5',
    ''
]

model_mapper = dict(zip(lst_models, lst_model_fname))


def run():
    # texts
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

    ### start demo
    st.markdown(
        """
        ## Test 
        """
    )

    # initialize session states to store all the options
    if "model" not in st.session_state:
        st.session_state['model'] = None
    if "label" not in st.session_state:
        st.session_state['label'] = 'Normal'
    if "use_mask" not in st.session_state:
        st.session_state['use_mask'] = None
    if "img_source" not in st.session_state:
        st.session_state['img_source'] = None
    if "img_idx" not in st.session_state:
        st.session_state['img_idx'] = 0
    if "img_title" not in st.session_state:
        st.session_state['img_title'] = ""
    if "img" not in st.session_state:
        st.session_state['img'] = None
    if "mask_source" not in st.session_state:
        st.session_state['mask_source'] = None
    if "img_show" not in st.session_state:
        st.session_state['img_show'] = False
    if "vgg_predict" not in st.session_state:
        st.session_state['vgg_predict'] = False
    if "grad_cam" not in st.session_state:
        st.session_state['grad_cam'] = False


    # choose model  # todo remove
    model_version_name = st.selectbox(
        "Chosir le modèle entrainé",
        (lst_models))
    model_fname = model_mapper[model_version_name]
    st.session_state['model'] = model_fname

    # choose label (?)  # todo put after source
    label_option = st.selectbox(
        "Choisir label de l'image",
        ('Covid', 'Normal', 'Viral Pneumonia', 'Lung Opacity'))
    label = OPT2LABEL_MAPPER[label_option]
    st.session_state.label = label

    # choose if use mask and how to use it
    use_mask = st.radio("selectionnez méthode de masque",
                        ('avec masque (recommendé)', 'sans masque', 'avec masque inversé'))
    st.session_state.use_mask = use_mask

    # choose image and mask sources
    col1, col2 = st.columns(2)
    with col1:
        image_source = st.radio(
            "Image source",
            ('aléatoirement de Kaggle dataset', 'télécharger votre radiographie')
        )
        st.session_state['img_source'] = image_source

        if st.session_state['img_source'] == 'télécharger votre radiographie':
            uploaded_file = st.file_uploader("Télécharge l'image")
            if uploaded_file is not None:
                # To read file as bytes:
                st.session_state['img'] = uploaded_file.getvalue()
                st.session_state['img_title'] = "votre image radio téléchargé"
        else:
            st.session_state['img_idx'] = random.choice(range(1, NUM_ALL_IMG[LABEL_MAPPER[label]] + 1))  # todo choose after 2000
            st.session_state['img'] = load_resize_img(label, st.session_state['img_idx'])
            img_title = label + '-' + str(st.session_state['img_idx'])
    if st.session_state.use_mask != 'sans masque':
        with col2:
            if st.session_state['img_source'] == 'télécharger votre radiographie':
                mask_source = st.radio(
                    "Mask source",
                    ('télécharger votre fichier de masque', 'généré par U-Net')
                )
            else:
                # choose mask source
                mask_source = st.radio(
                    "Mask source",
                    ('De Kaggle dataset', 'généré par U-Net')
                )
        if mask_source == "généré par U-Net":
            inverse = (st.session_state.use_mask == 'avec masque inversé')
            mask, X = unet_pipeline(st.session_state['img'], inverse)
        else:
            if mask_source == 'De Kaggle dataset':
                mask = load_mask(label, st.session_state['img_idx'] )
            elif mask_source == 'télécharger votre fichier de masque':
                uploaded_masque = st.file_uploader("Télécharge le masque")
                if uploaded_masque is not None:
                    # To read file as bytes:
                    mask = uploaded_masque.getvalue()
            if st.session_state.use_mask == 'avec masque inversé':
                mask = 1 - mask
            X = st.session_state['img'] * mask  # application of mask
    else:
        X = st.session_state['img']

    # show image, mask and masked image
    if st.button("afficher l'image (et masque)"):
        st.session_state['img_show'] = True
    if st.session_state['img_show']:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(131)
        imshow(st.session_state['img'])
        plt.title(img_title)
        plt.axis('off')
        if st.session_state.use_mask != "sans masque":
            plt.subplot(132)
            imshow(mask)
            plt.title("masque")
            plt.axis('off')
            plt.subplot(133)
            imshow(X)
            plt.title("image masqué")
            plt.axis('off')
        st.pyplot(fig)

    # load VGG model and predict label
    if st.button("load le modèle entrainé et obtenir la prédiction"):
        st.session_state['vgg_predict'] = True
    if st.session_state['vgg_predict']:
        y_pred, res_grad_cam = vgg_prediction(model_fname, X, grad_cam=True)
        st.text(str(y_pred))

    # show grad cam result
    if st.button("show Grad-CAM result"):
        st.session_state['grad_cam'] = True
    if st.session_state['grad_cam']:
        fig = plt.figure(figsize=(5, 5))
        imshow(res_grad_cam)  # todo erreur maybe opacity, check the google cv2 function how it adjust opacity
        st.pyplot(fig)



