# todo complete with the right code and all the models

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.morphology import disk
from scipy import ndimage as nd
import cv2

# authorize access to google drive
from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)


# set paths
DIR_DATA = 'projet_radio_covid/COVID-19_Radiography_Dataset/'  # set to local path
DIR_OUTPUT = 'projet_radio_covid'

LST_GROUP = ['covid', 'normal', 'viral', 'opac']
LST_FOLDERS = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']
FNAME_MAPPER = dict(zip(LST_GROUP, LST_FOLDERS))
LABEL_MAPPER = dict(zip(LST_GROUP, range(4)))

DIRS = dict(zip(LST_GROUP, [os.path.join(DIR_DATA, folder) for folder in LST_FOLDERS]))
DIR_IMAGES = dict(zip(LST_GROUP, [os.path.join(dir, 'images') for dir in DIRS.values()]))
DIR_MASKS = dict(zip(LST_GROUP, [os.path.join(dir, 'masks') for dir in DIRS.values()]))

NUM_ALL_IMG = [3616, 10192, 1345, 6012]

def get_fname(group, idx):
  return(f"{FNAME_MAPPER[group]}-{idx}.png")


def crop_black_border(image):
  assert image.shape[0] == image.shape[1]
  old_dim = image.shape[0]

  mask = image!=0
  mask_row = mask.any(0)
  mask_col = mask.any(1)

  row_range = old_dim - mask_row.argmax() - mask_row[::-1].argmax()
  col_range = old_dim - mask_col.argmax() - mask_col[::-1].argmax()

  if row_range < col_range:
    top = mask_col.argmax()
    bottom = old_dim - mask_col[::-1].argmax()
    left = mask_row.argmax() - (col_range - row_range)//2
    right = old_dim - mask_row[::-1].argmax() + (col_range - row_range)//2
    if left < 0:
      left, right = 0, col_range
    elif right > old_dim:
      left, right = old_dim - col_range, col_range
  else:
    left = mask_row.argmax()
    right = old_dim - mask_row[::-1].argmax()
    top = mask_col.argmax() - (row_range - col_range)//2
    bottom = old_dim - mask_col[::-1].argmax() + (row_range - col_range)//2
    if top < 0:
      top, bottom = 0, row_range
    elif bottom > old_dim:
      top, bottom = old_dim - row_range, old_dim
  return image[top:bottom, left:right]


def normalize_intensity(img, mean=.2, min=0):
  return(img/img.mean() * (mean - min) + min)


from itertools import compress, product


def preprocessing_pipeline(path_image: list, path_mask=None, image_size=(50, 50), mask=True,
                           crop=True, normalize=True, flatten=True, path_output=None, **kwargs):
    # image_size_50_50_mask_crop
    # image_size_50_50_mask_crop_normalize_02_0
    # image_size_100_100_mask_crop
    # image_size_100_100_mask_crop_normalize_02_0

    # image_size_50_50_mask_crop_normalize_02_0_flatten_pca_dim_10
    # image_size_50_50_mask_crop_normalize_02_0_flatten_pca_dim_09
    # image_size_100_100_mask_crop_normalize_02_0_flatten_pca_dim_09
    # image_size_50_50_mask_crop_flatten_pca_dim_10
    # image_size_50_50_mask_crop_flatten_pca_dim_09
    # image_size_100_100_mask_crop_flatten_pca_dim_10
    # image_size_100_100_mask_crop_flatten_pca_dim_09
    """
    Parameters
    ------------
    path_image: list of paths to images.
    image_size: tuple: the destinated size of each image. If none, the image will be 256*256.
    mask: bool: whether apply the masks to the radio images.
    crop: bool: whether crop the black border after applying mask.
    flatten: bool: return the images as matrix if False; else, return flattened vectors.

    Return:
    ------------
    X: list of arrays or vectors
    y: array of labels (value can be 1, 2, 3, 4)

    """
    if path_output is not None:
        assert len(path_image) == len(path_output)
        check_exist = [os.path.exists(path) for path in path_output]
        if sum(check_exist):
            print("files already exist:", list(compress(path_output, check_exist)))
            filter_exist = [not _ for _ in check_exist]
            path_output = list(compress(path_output, filter_exist))
            path_image = list(compress(path_image, filter_exist))
            if path_mask is not None:
                path_mask = list(compress(path_mask, filter_exist))
            else:
                pass
        else:
            pass

    print("making files:", path_output)

    # load the images
    print('image loading ...')
    X = np.array([resize(imread(path, as_gray=True), (256, 256), anti_aliasing=True) for path in path_image])

    # masking
    if mask:
        print('mask loading ...')
        masks = np.array([imread(path, as_gray=True) for path in path_mask])
        X = X * masks

    # crop the black border and resize
    if crop:
        print('border cropping ...')
        X = np.array([resize(crop_black_border(image), image_size, anti_aliasing=True) for image in X])
    else:
        print('resizing ...')
        X = resize(X, (X.shape[0], *image_size), anti_aliasing=True)

    # normalize intensity
    if normalize:
        min_intensity = kwargs.get('min', 0)
        mean_intensity = kwargs.get('mean', .2)
        X = normalize_intensity(X, mean=mean_intensity, min=min_intensity)  # todo use min_max as option

    if not flatten:
        if path_output is not None:
            for i, path in enumerate(path_output):
                plt.imsave(path, X[i], cmap='gray')
        else:
            return X
    else:
        # reshape each image to 1-D vector
        X = X.reshape(X.shape[0], -1)
        if path_output is not None:
            for i, path in enumerate(path_output):
                with open(path, 'w') as fo:
                    json.dump(X[i].tolist(), fo)
        else:
            return X


# Images are unmasked
# NUMBER OF IMAGES
diviseur = 4 # as 25%
fraction = [x//diviseur for x in NUM_ALL_IMG]

# SIZE OF IMAGE
imgx = 50
imgy = 50

X_train, X_test, y_train, y_test = preprocessing_pipeline(
    groups=LST_GROUP,
    num_images=fraction,   # pass NUM_ALL_IMG to use all the images.
    image_size=(imgx, imgy),
    use_mask=False,
    crop=False,
    output_type='matrix',
    )


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print('\n', X_train.shape, y_train.shape, X_test.shape, y_test.shape)


### 1.1  LOGISTIC REGRESSION ON 25% images per class
# resize 50, mask = NO, normalze = NO, PCA = No
# Regression Params :  solver = saga, C = 0.1, max_iter = 2000
# other log solvers = lbfgs, liblinear, newton_cg, newton_cholesky, sag, sag

log_reg = LogisticRegression(C=0.1, solver = 'saga', max_iter = 3000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Logistic regression on unreduced data :", log_reg.score(X_test, y_test), '\n')
display(pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
print('\n',classification_report_imbalanced(y_test, y_pred))


probs = log_reg.predict_proba(X_test)
#y_pred = svc.predict(X_test_pca)

y_test_dic =pd.get_dummies(y_test)
#y_train_dic=pd.get_dummies(y_train)

from sklearn.metrics import confusion_matrix

#
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show();


# dimension reduction by applying PCA (90% of the explained variance)
# images are still unmasked and uncrop
pca = PCA(n_components = .9)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(pca.n_components_)


# Display of 36 PCA main components
plt.figure(figsize=(8, 7))
for i in range(36):
    plt.subplot(7,6,i+1)
    plt.imshow(pca.components_[i].reshape(50,50), cmap='gray')
    plt.axis("off")
plt.show();


log_reg = LogisticRegression(C=0.1, solver = 'saga', max_iter = 3000)
log_reg.fit(X_train_pca, y_train)

y_pred = log_reg.predict(X_test_pca)

probs = log_reg.predict_proba(X_test_pca)
#y_pred = svc.predict(X_test_pca)

y_test_dic =pd.get_dummies(y_test)
#y_train_dic=pd.get_dummies(y_train)