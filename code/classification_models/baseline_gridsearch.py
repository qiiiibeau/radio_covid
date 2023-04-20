## todo add the following models: lr, svm

import os

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble

from preprocessing import dataload_preprocessing
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
X_train, X_test, y_train, y_test, args = dataload_preprocessing(
    groups=lst_group,
    num_images=np.array(NUM_ALL_IMG) // 10,  # pass NUM_ALL_IMG to use all the images.
    image_size=(50, 50),
    use_mask=True,
    crop=True,
    output_type='vector',
    dim=0.9,
    normalize=False
)

# logistic regression
logreg = LogisticRegression()
param_grid = {'C': [0.1, 1, 10, 100],
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear', 'saga', 'lbfgs'],
              'multi_class': ['ovr', 'multinomial'],
              'max_iter': [100, 1000, 2000]
              }
grid_search = GridSearchCV(logreg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best hyperparameters found for logistic regression: ", grid_search.best_params_)

# svm
svm = SVC()
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'gamma': ['scale', 'auto', 0.1, 1, 10],
              'class_weight': [None, 'balanced']
              }
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best hyperparameters found for svm: ", grid_search.best_params_)

# random forest
rfc = ensemble.RandomForestClassifier(n_jobs=-1, random_state=23)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best hyperparameters found for random forest: ", grid_search.best_params_)

# gradient boosting
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
print("Best hyperparameters found for gradient boosting: ", grid_search.best_params_)
