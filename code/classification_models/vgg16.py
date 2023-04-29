## v9  vgg16 unfreezed, data class are balanced

import itertools
from matplotlib import cm
import matplotlib.pyplot as plt

from imblearn.metrics import classification_report_imbalanced

from keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from timeit import default_timer as timer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16

from utils import *
from preprocessing import *

# image size
x_img = 200
y_img = 200

n_covid = 2000
n_normal = 2000
n_viral = 1345
n_opac = 2000

X_train, X_test, y_train, y_test = dataload_preprocessing(
    groups=LST_GROUP,
    num_images=[n_covid, n_normal, n_viral, n_opac],
    image_size=(x_img, y_img),
    use_mask=True,
    crop=True,
    output_type='matrix',
)

X_train = X_train.reshape(-1, x_img, y_img, 1)
X_test = X_test.reshape(-1, x_img, y_img, 1)

print('\n', X_train.shape, X_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\n', y_train.shape, y_test.shape)
print(y_test[0])


# callbacks
class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


model_checkpoint_cnn = ModelCheckpoint(filepath='projet_radio_covid/callbacks_cnn',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.005,
                               patience=5,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         min_delta=0.002,
                                         patience=3,
                                         factor=0.1,
                                         cooldown=4,
                                         verbose=1)

time_callback = TimingCallback()

# Unfreezed VGG16
X_train = X_train.reshape(-1, x_img, y_img, 1)
X_test = X_test.reshape(-1, x_img, y_img, 1)

print('\n', X_train.shape, X_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\n', y_train.shape, y_test.shape)
print(y_test[0])

n_class = 4

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(x_img, y_img, 3))

for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Sequential()
model.add(base_model)

model.add(GlobalAveragePooling2D())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=n_class, activation='softmax'))

print(model.summary())

model_checkpoint_uvgg16 = ModelCheckpoint(filepath='projet_radio_covid/callbacks_uvgg16',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='min')

model.load_weights('projet_radio_covid/callbacks_uvgg16')

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=1e-4),
              metrics=["accuracy"])

batch_size = 200

history_uvgg16 = model.fit(X_train, y_train,
                           epochs=50,
                           steps_per_epoch=X_train.shape[0] // batch_size,
                           validation_split=0.2,
                           callbacks=[model_checkpoint_uvgg16, early_stopping, reduce_learning_rate, time_callback],
                           verbose=True)

test_pred_uvgg16 = model.predict(X_test)

test_pred_uvgg16_class = np.argmax(test_pred_uvgg16, axis=1)
y_test_class = np.argmax(y_test, axis=1)

save_pred(y_test_class, test_pred_uvgg16_class, 'uvgg16', "class_balanced", clf_params="")  # todo save params
save_metrics(y_test_class, test_pred_uvgg16_class, 'uvgg16', "class_balanced", clf_params="")

# model evaluation
print(classification_report_imbalanced(y_test_class, test_pred_uvgg16_class))
cnf_matrix = confusion_matrix(y_test_class, test_pred_uvgg16_class)
### Display a confusion matrix rate as a coloured table

cnf_rescaled = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        cnf_rescaled[i, j] = cnf_matrix[i, j] / cnf_matrix[i, :].sum()

classes = range(0, 4)
plt.figure(figsize=(3, 3))

plt.imshow(cnf_rescaled, interpolation='nearest', cmap='Blues')
plt.title("Matrice de confusion %")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
    plt.text(j, i, round(cnf_rescaled[i, j], 2),
             horizontalalignment="center",
             color="white" if cnf_rescaled[i, j] > (cnf_rescaled.max() / 2) else "black")

plt.ylabel('Vrais labels')
plt.xlabel('Labels prédits')
plt.show()

# classes that are most often confused
# Error thresould = 10%

threshold = 0.1
for i in range(4):
    for j in range(4):
        if ((cnf_rescaled[i, j] > threshold) and (j != i)):
            print('la classe {} est confondue avec la classe {}'.format(i, j))

error_indexes = []
for i in range(len(test_pred_uvgg16)):
    if (test_pred_uvgg16_class[i] != y_test_class[i]):
        if (test_pred_uvgg16_class[i] != y_test_class[i]):
            error_indexes += [i]

j = 1
for i in np.random.choice(error_indexes, size=3):
    img = X_test[i]
    img = img.reshape(x_img, y_img)

    plt.subplot(1, 3, j)
    j = j + 1
    plt.axis('off')
    plt.imshow(img, cmap=cm.binary, interpolation='None')
    plt.title('True Label: ' + str(y_test_class[i]) \
              + '\n' + 'Prediction: ' + str(test_pred_uvgg16_class[i]) \
              + '\n' + 'Confidence: ' + str(round(test_pred_uvgg16[i][test_pred_uvgg16_class[i]], 2)))
