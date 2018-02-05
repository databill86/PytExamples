# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

# Initialising the CNN
classifier = Sequential() # initializing of NN as a sequence of layers

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#   32 - filters - number of feature detectors, (3,3) - kernel size - size of feature detector
#   input_shape (64,64, 3)
#       64, 64 dimension of input array (picture) - all images will be preprocessed to have these dimensions
#       3 - number of channels (R,G,B)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#   pool_size (2,2) - size of matrix to use for pooling, it reduces size of input

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) # rule of thumb to choose number of hidden nodes between number of input and output nodes, and power of 2
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1, activation = 'sigmoid')) # sigmoid - binary outcome, more classes - softmax

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# aroung 90% accuracy on test set https://www.udemy.com/deeplearning/learn/v4/questions/2276518

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
# performs image augmentation to prevent overffitting, it creates random batches with random pictures,
# which are flipped, modified in some way, rotating, shifting, randomly transformed, etc.
# it enriches image dataset and allows to get good performance with less number of images
# low probability to find same image in different batches

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) # image data random transformations


test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

print(training_set.class_indices) # indexes of predicted classes

# target size - dimensions of output image - are used as input parameters for NN
# class mode - binary (2 classes)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 2000/32)
# steps per epoch - number of training images
# validation steps - number of test samples

classifier.save('dog_cats_model1.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# ----------------------------------- predict one picture -------------------------------

from keras.models import load_model
import cv2
import numpy as np

dog_cat_model = load_model('dog_cats_model1.h5')

# https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image/43019294

img_dog = cv2.imread('some_dog.jpg')
img_dog = cv2.resize(img_dog, (64,64))
img_dog = img_dog * (1./255)
img_dog = np.reshape(img_dog, [1, 64, 64, 3])
print(dog_cat_model.predict(img_dog))

img_cat = cv2.imread('some_cat.jpg')
img_cat = cv2.resize(img_cat, (64,64))
img_cat = img_cat * (1./255)
img_cat = np.reshape(img_cat, [1, 64, 64, 3])
print(dog_cat_model.predict(img_cat))

# training_set.class_indices - 1 - dog