## v9

import itertools
from matplotlib import cm
import matplotlib.pyplot as plt

from imblearn.metrics import classification_report_imbalanced

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from timeit import default_timer as timer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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

# CNN model
inputs = Input(shape=(x_img, y_img, 1), name="Input")
first_layer = Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', name="first_layer")
second_layer = MaxPooling2D(pool_size=(2, 2), name="second_layer")
third_layer = Dropout(rate=0.2, name='third_layer')
fourth_layer = Flatten()  # pour passer les matrice en vecteurs
fifth_layer = Dense(units=128, activation="relu", name="fifth_layer")
output_layer = Dense(units=4, activation="softmax", name="output_layer")

x = first_layer(inputs)
x = second_layer(x)
x = third_layer(x)
x = fourth_layer(x)
x = fifth_layer(x)
outputs = output_layer(x)

cnn = Model(inputs=inputs, outputs=outputs)
print(cnn.summary())

cnn.compile(loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])

batch_size = 200
training_history_cnn = cnn.fit(X_train, y_train,
                               epochs=50, steps_per_epoch=X_train.shape[0] // batch_size,
                               validation_split=0.2,
                               callbacks=[model_checkpoint_cnn, early_stopping, reduce_learning_rate, time_callback],
                               verbose=True)

# model evaluation
train_loss_cnn = training_history_cnn.history['loss']
val_loss_cnn = training_history_cnn.history['val_loss']

test_pred_cnn = cnn.predict(X_test)

test_pred_cnn_class = np.argmax(test_pred_cnn, axis=1)
y_test_class = np.argmax(y_test, axis=1)

save_pred(y_test_class, test_pred_cnn_class, 'cnn', "", clf_params="")  # todo save params
save_metrics(y_test_class, test_pred_cnn_class, 'cnn', "", clf_params="")


print(classification_report_imbalanced(y_test_class, test_pred_cnn_class))

cnf_matrix = confusion_matrix(y_test_class, test_pred_cnn_class)
### Display a confusion matrix rate (because of imbalanced sample) as a coloured table

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

# Seuil d'erreur = 8%
threshold = 0.08
for i in range(4):
    for j in range(4):
        if ((cnf_rescaled[i, j] > threshold) and (j != i)):
            print('la classe {} est confondue avec la classe {}'.format(i, j))

# Errors between 'covid' (0) and 'normal' (1)
error_indexes = []
for i in range(len(test_pred_cnn)):
    if (test_pred_cnn_class[i] != y_test_class[i]):
        if (y_test_class[i] == 0 or y_test_class[i] == 1):
            if (test_pred_cnn_class[i] == 0 or test_pred_cnn_class[i] == 1):
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
              + '\n' + 'Prediction: ' + str(test_pred_cnn_class[i]) \
              + '\n' + 'Confidence: ' + str(round(test_pred_cnn[i][test_pred_cnn_class[i]], 2)))
