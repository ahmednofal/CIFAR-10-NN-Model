import numpy as np
from keras import optimizers

#np.random.seed(1337)


from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU,ThresholdedReLU
from keras.activations import relu, tanh, elu
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.constraints import maxnorm

from keras.callbacks import ModelCheckpoint, TensorBoard

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

nb_classes = 10
#reshaping
shift = 0

model = ImageDataGenerator(featurewise_center= True,
 samplewise_center=False,
 featurewise_std_normalization=True,
 samplewise_std_normalization=False,
 zca_whitening=True,
 rotation_range=90,
 width_shift_range=shift,
 height_shift_range=shift,
 horizontal_flip=False,
 vertical_flip=False,
 )


model.fit(X_train)

X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()


model = Sequential()
model.add(Dense(1000,
kernel_constraint=maxnorm(5),
input_shape = X_train.shape[1:],
bias_initializer='zeros'
))

model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.4))

model.add(Dense(1500,
kernel_constraint=maxnorm(4)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.3))

#,kernel_regularizer = regularizers.l2(0.01)
model.add(Dense(1500,
kernel_constraint=maxnorm(4)))
model.add(BatchNormalization())
model.add(ELU(alpha = 4e-5))
model.add(Dropout(0.2))

model.add(Dense(10,
kernel_constraint=maxnorm(4)))
model.add(BatchNormalization())
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.0899, momentum=0.95, decay=1e-2, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="/home/naufal/Downloads/CIFAR10-NN-master/CIFAR-10-NN-saved-output/tmp/run3/weights.h5", verbose=0   , save_best_only=True)
graph_save = TensorBoard(log_dir='/home/naufal/Downloads/CIFAR10-NN-master/CIFAR-10-NN-saved-output/tmp/run3', histogram_freq=50, write_graph=True, write_images=False)
model.fit(X_train, Y_train,epochs=1000,batch_size=2048, validation_data = (X_test, Y_test), callbacks=[checkpointer,graph_save])


