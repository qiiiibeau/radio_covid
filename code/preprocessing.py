import os
import json

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize

from utils import DIR_DATA, DIR_IMAGES, DIR_MASKS, LST_GROUP, LABEL_MAPPER, NUM_ALL_IMG, get_fname


# preprocessing
def crop_black_border(image):
    """
    crop the black borders of an image after application of mask.
    :param image: ndarray, image after application of mask
    :return: ndarray, image after croping
    """
    assert image.shape[0] == image.shape[1]
    old_dim = image.shape[0]

    mask = image != 0
    mask_row = mask.any(0)
    mask_col = mask.any(1)

    row_range = old_dim - mask_row.argmax() - mask_row[::-1].argmax()
    col_range = old_dim - mask_col.argmax() - mask_col[::-1].argmax()

    if row_range < col_range:
        top = mask_col.argmax()
        bottom = old_dim - mask_col[::-1].argmax()
        left = mask_row.argmax() - (col_range - row_range) // 2
        right = old_dim - mask_row[::-1].argmax() + (col_range - row_range) // 2
        if left < 0:
            left, right = 0, col_range
        elif right > old_dim:
            left, right = old_dim - col_range, col_range
    else:
        left = mask_row.argmax()
        right = old_dim - mask_row[::-1].argmax()
        top = mask_col.argmax() - (row_range - col_range) // 2
        bottom = old_dim - mask_col[::-1].argmax() + (row_range - col_range) // 2
        if top < 0:
            top, bottom = 0, row_range
        elif bottom > old_dim:
            top, bottom = old_dim - row_range, old_dim
    return image[top:bottom, left:right]


def normalize_intensity(img, mean=.2, min=0):
    return (img / img.mean() * (mean - min) + min)


# image loading and preprocessing
def dataload_preprocessing(groups=list, num_images=list, image_size=(50, 50), use_mask=True,
                           crop=True, flatten=False, normalize=False, path_output=None, **kwargs):
    """
    Preprocessing pipeline

    Parameters
    ------------
    groups: list of categories (labels) of input data
    num_images: list of number of images in each group. pass NUM_ALL_IMG to use all the images in current dataset.
    image_size: tuple: the destinated size of each image. If none, the image will be 256*256.
    use_mask: bool: whether apply the masks to the radio images.
    crop: bool: whether crop the black border after applying mask.
    flatten: bool: return the images as matrix if False; else, return flattened vectors.

    Return:
    ------------
    X, y

    """
    X = np.zeros((sum(num_images), 256, 256))
    y = np.zeros(sum(num_images))
    sum_num = 0

    # load the images
    print('image loading ...')
    for group, num in zip(groups, num_images):
        X[sum_num:(sum_num + num), :, :] = [
            resize(imread(os.path.join(DIR_IMAGES[group], get_fname(group, idx)), as_gray=True), (256, 256),
                   anti_aliasing=True)
            for idx in range(1, num + 1)]
        y[sum_num:sum_num + num] = [LABEL_MAPPER[group]] * num
        sum_num += num

    # masking
    if use_mask:
        print('mask loading ...')
        masks = np.zeros((sum(num_images), 256, 256))
        sum_num = 0
        for group, num in zip(groups, num_images):
            masks[sum_num:(sum_num + num), :, :] = [
                imread(os.path.join(DIR_MASKS[group], get_fname(group, idx)), as_gray=True)
                for idx in range(1, num + 1)]
            sum_num += num
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
            return X, y
    else:
        # reshape each image to 1-D vector
        X = X.reshape(X.shape[0], -1)
        if path_output is not None:
            for i, path in enumerate(path_output):
                with open(path, 'w') as fo:
                    json.dump(X[i].tolist(), fo)
        else:
            return X, y


# image loading : load already preprocessed data directly (for data in drive)
def load_preprocessed_data(params):
    num_images = np.array(NUM_ALL_IMG) // params['frac']
    X = np.zeros((sum(num_images), *params['image_size']))
    y = np.zeros(sum(num_images))
    sum_num = 0

    # load the images
    print('preprocessed image loading ...')
    for group, num in zip(LST_GROUP, num_images):
        X[sum_num:(sum_num + num), :, :] = [
            imread(os.path.join(DIR_DATA, 'image_size_128_128_mask_crop', group + '_' + str(idx) + '.png'),
                   as_gray=True)  ## todo use param
            for idx in range(1, num + 1)]
        y[sum_num:sum_num + num] = [LABEL_MAPPER[group]] * num
        sum_num += num
    return X, y

