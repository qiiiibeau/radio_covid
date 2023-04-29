### v9

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

# LeNet5 model
inputs = Input(shape=(x_img, y_img, 1))

conv_1 = Conv2D(filters=30,
                input_shape=(x_img, y_img, 1),
                kernel_size=(5, 5),
                padding='valid',
                activation='relu')

maxpool_1 = MaxPooling2D(pool_size=(2, 2))

conv_2 = Conv2D(filters=16,
                kernel_size=(3, 3),
                padding='valid',
                activation='relu')

maxpool_2 = MaxPooling2D(pool_size=(2, 2))
dropout = Dropout(rate=0.2)
flatten = Flatten()
dense_1 = Dense(units=128, activation="relu")
dense_2 = Dense(units=4, activation="softmax")

x = conv_1(inputs)
x = maxpool_1(x)
x = conv_2(x)
x = maxpool_2(x)
x = dropout(x)
x = flatten(x)
x = dense_1(x)
outputs = dense_2(x)

lenet = Model(inputs=inputs, outputs=outputs)
print(lenet.summary())

model_checkpoint_lenet = ModelCheckpoint(filepath='projet_radio_covid/callbacks_lenet',
                                         monitor='val_loss',
                                         save_best_only=True,
                                         mode='min')

lenet.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

batch_size = 200
training_history_lenet = lenet.fit(X_train, y_train,
                                   epochs=50, steps_per_epoch=X_train.shape[0] // batch_size,
                                   validation_split=0.2,
                                   callbacks=[model_checkpoint_lenet, early_stopping, reduce_learning_rate,
                                              time_callback],
                                   verbose=True)

# model evaluation
test_pred_lenet = lenet.predict(X_test)

test_pred_lenet_class = np.argmax(test_pred_lenet, axis=1)
y_test_class = np.argmax(y_test, axis=1)

save_pred(y_test_class, test_pred_lenet_class, 'lenet', "", clf_params="")  # todo save params
save_metrics(y_test_class, test_pred_lenet_class, 'lenet', "", clf_params="")


print(classification_report_imbalanced(y_test_class, test_pred_lenet_class))
cnf_matrix = confusion_matrix(y_test_class, test_pred_lenet_class)

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
# Error thresould = 7%

threshold = 0.07
for i in range(4):
    for j in range(4):
        if ((cnf_rescaled[i, j] > threshold) and (j != i)):
            print('la classe {} est confondue avec la classe {}'.format(i, j))

error_indexes = []
for i in range(len(test_pred_lenet)):
    if (test_pred_lenet_class[i] != y_test_class[i]):
        if (test_pred_lenet_class[i] != y_test_class[i]):
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
              + '\n' + 'Prediction: ' + str(test_pred_lenet_class[i]) \
              + '\n' + 'Confidence: ' + str(round(test_pred_lenet[i][test_pred_lenet_class[i]], 2)))


