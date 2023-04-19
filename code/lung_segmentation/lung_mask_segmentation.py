

import os
import numpy as np

from lungmask import mask
import SimpleITK as sitk
from skimage.io import imread

import matplotlib.pyplot as plt


DIR_DATA_ORIGINAL = 'data/COVID-19_Radiography_Dataset/'
INPUT = os.path.join(DIR_DATA_ORIGINAL, 'Normal/images/Normal-3.png')


np_img_2d = imread(INPUT)
plt.imshow(np_img_2d)
print(np_img_2d.ndim)
print(np_img_2d.shape)
np_img_3d = np.zeros((3, 299, 299))

for i in range(3):
    for j in range(299):
        for k in range(299):
            np_img_3d[i, j, k] = np_img_2d[j, k]

print(np_img_3d.ndim)

print(np_img_3d.shape)

model = mask.get_model('unet', 'R231CovidWeb')
# segmentation = mask.apply(input_image, batch_size=1)  # default model is U-net(R231)
segmentation = mask.apply(np_img_3d, batch_size=1, model=model)  # default model is U-net(R231)

