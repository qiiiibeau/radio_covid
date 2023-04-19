import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from code.utils import get_fname, get_str_param, get_dir_pred, get_path_metric, load_summary_metrics, save_pred, \
    save_metrics
from code.preprocessing import dataload_preprocessing, load_preprocessed_data

# set paths
DIR_DATA_LOCAL = 'projet_radio_covid/COVID-19_Radiography_Dataset/'  # set to local path
DIR_DATA_GDRIVE = 'projet_radio_covid/data/'

LST_GROUP = ['covid', 'normal', 'viral', 'opac']
LST_FOLDERS = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']
FNAME_MAPPER = dict(zip(LST_GROUP, LST_FOLDERS))
LABEL_MAPPER = dict(zip(LST_GROUP, range(4)))

DIRS = dict(zip(LST_GROUP, [os.path.join(DIR_DATA_LOCAL, folder) for folder in LST_FOLDERS]))
DIR_IMAGES = dict(zip(LST_GROUP, [os.path.join(dir, 'images') for dir in DIRS.values()]))
DIR_MASKS = dict(zip(LST_GROUP, [os.path.join(dir, 'masks') for dir in DIRS.values()]))

NUM_ALL_IMG = [3616, 10192, 1345, 6012]

# read and preprocess the images

# images with masks
baseline_prepro_params = {
    'frac': 4,
    'image_size': (128, 128),
    'mask': True,
    'crop': True,
    'flatten': True,
    'normalize': False,
    'pca': 0.9
}

# load already preprocessed data directly
t0 = time.time()
X, y = load_preprocessed_data(baseline_prepro_params)
print("loading time", round(time.time() - t0), 's, i.e.', round((time.time() - t0) / 60), 'min')

if baseline_prepro_params['flatten']:
    X = X.reshape(X.shape[0], -1)

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, shuffle=True, stratify=y)

# pca
if baseline_prepro_params['pca'] is not None:
    t0 = time.time()
    pca = PCA(n_components=baseline_prepro_params['pca'])
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print("pca time", round(time.time() - t0), 's, i.e.', round((time.time() - t0) / 60), 'min')

# Logistic Regression

clf_name = "lr"

lr_params = {'multi_class': 'multinomial', 'solver': 'lbfgs', 'penalty': 'l2', 'C': 1.0, 'max_iter': 2000}

logreg = LogisticRegression()
logreg.set_params(**lr_params)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
save_pred(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=lr_params)
save_metrics(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=lr_params)

# SVM
clf_name = 'svm'

svc_params = {"C": 10, 'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced'}

svc = SVC()  # C=1 par défaut, kernel=rbf' par défaut, gamma='scale' par défaut (sauf pour linear)
svc.set_params(**svc_params)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

clf_params = {'kernel': 'linear'}
save_pred(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=svc_params)
save_metrics(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=svc_params)

# random forest
clf_name = "rf"

rf_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
rf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=23)
rf.set_params(**rf_params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

save_pred(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=rf_params)
save_metrics(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=rf_params)

# Gradient Boosting

clf_name = "gb"

gb = ensemble.GradientBoostingClassifier()
gb_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2,
             'n_estimators': 300}
gb.set_params(**gb_params)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

save_pred(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=gb_params)
save_metrics(y_test, y_pred, clf_name, baseline_prepro_params, clf_params=gb_params)
