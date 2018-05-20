from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('/home/iblutman155201/dog-project/dogImages/train')
valid_files, valid_targets = load_dataset('/home/iblutman155201/dog-project/dogImages/valid')
test_files, test_targets = load_dataset('/home/iblutman155201/dog-project/dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/home/iblutman155201/dog-project/dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("/home/iblutman155201/dog-project/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


import cv2
import matplotlib.pyplot as plt
# % matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('/home/iblutman155201/dog-project/haarcascades/haarcascade_frontalface_alt.xml')


def show_output(img_path):
    # load color (BGR) image
    img = cv2.imread(img_path)
    # display the image, along with bounding box
    plt.imshow(img)
    plt.show()


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

### Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('/home/iblutman155201/dog-project/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

def create_DogResnet50_model():
    from keras.models import Sequential
    from keras.layers import GlobalAveragePooling2D, Dense
    ### Define your architecture.
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    model.add(Dense(133, activation='softmax'))
    ### Compile the model.
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train_DogResnet50_model(model):
    from keras.callbacks import ModelCheckpoint
    ### Train the model.
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                                   verbose=1, save_best_only=True)
    model.fit(train_Resnet50, train_targets,
                       validation_data=(valid_Resnet50, valid_targets),
                       epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


def test_DogResnet50_model(model):
    ### Calculate classification accuracy on the test dataset.
    # get index of predicted dog breed for each image in test set
    DogResnet50_predictions = [np.argmax(DogResnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in
                               test_Resnet50]
    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(DogResnet50_predictions) == np.argmax(test_targets, axis=1)) / len(
        DogResnet50_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


### Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def DogResnet50_predict_breed(img_path):
    from extract_bottleneck_features import extract_Resnet50
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = DogResnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def classifier(img_path):
    if dog_detector(img_path):
        print('Hello, human!')
        print(DogResnet50_predict_breed(img_path))
    elif face_detector(img_path):
        print('Hello, Dog!')
        print(DogResnet50_predict_breed(img_path))
    else:
        print('Error - neither dog nor human detected in the image')


DogResnet50_model = create_DogResnet50_model()
DogResnet50_model.summary()
# train_DogResnet50_model(DogResnet50_model)

### Load the model weights with the best validation loss.
DogResnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
# test_DogResnet50_model(DogResnet50_model)

# dog_name = DogResnet50_predict_breed(test_files[0])

classifier(human_files[0])
classifier(test_files[0])
