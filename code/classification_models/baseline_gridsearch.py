## todo add the following models: lr, svm

import os

import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn.metrics import classification_report
import xgboost as xgb

from code.preprocessing import dataload_preprocessing
from sklearn.model_selection import GridSearchCV

# set paths
dir_data = '../../data/COVID-19_Radiography_Dataset/'  # set to local path
DIR_OUTPUT = '../result/baseline'

lst_group = ['covid', 'normal', 'viral', 'opac']
lst_folders = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']
mapper_fname = dict(zip(lst_group, lst_folders))
label_mapper = dict(zip(lst_group, range(4)))

DIRS = dict(zip(lst_group, [os.path.join(dir_data, folder) for folder in lst_folders]))
DIR_IMAGES = dict(zip(lst_group, [os.path.join(dir, 'images') for dir in DIRS.values()]))
DIR_MASKS = dict(zip(lst_group, [os.path.join(dir, 'masks') for dir in DIRS.values()]))

NUM_ALL_IMG = [3616, 10192, 1345, 6012]

# use 10% of the data
X_train, X_test, y_train, y_test, args = pipeline(
    groups=lst_group,
    num_images=np.array(NUM_ALL_IMG) // 10,   # pass NUM_ALL_IMG to use all the images.
    # num_images=[10, 10, 10, 10],   # pass NUM_ALL_IMG to use all the images.
    image_size=(50, 50),
    use_mask=True,
    crop=True,
    output_type='vector',
    dim=0.9,
    normalize=False
    )


df_metric = pd.DataFrame(columns=['preprocessing_param', 'model', 'model_param'] +
                                 ['recall_'+label for label in lst_group] +
                                 ['precision_' + label for label in lst_group] +
                                 ['f1_' + label for label in lst_group]
                         )

rfc = ensemble.RandomForestClassifier(n_jobs=-1, random_state=23)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create a GridSearchCV object with the parameter grid and the model
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the GridSearchCV object with your data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters found for random forest: ", grid_search.best_params_)

# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# print('random forest result')
# print(pd.crosstab(y_test, y_pred, rownames=['real class'], colnames=['predicted class']))
# res = classification_report(y_test, y_pred, target_names=lst_group)
# print(res)


gbc = ensemble.GradientBoostingClassifier()

param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(gbc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best hyperparameters found for gra: ", grid_search.best_params_)



# gbc.fit(X_train, y_train)
# y_pred = gbc.predict(X_test)
#
# print(pd.crosstab(y_pred, pd.Series(y_test)))
# print(classification_report(y_test, y_pred, target_names=lst_group))


#
# # train by batch  todo
# clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=23)
#
# for X_train, X_test, y_train, y_test in pipeline_batch(
#         num_batch=10,
#         groups=lst_group,
#         image_size=(50, 50),
#         use_mask=True,
#         crop=True,
#         output_type='vector',
#         dim=10,
#         normalize=True
# ):
#     clf.fit(X_train, y_train)

