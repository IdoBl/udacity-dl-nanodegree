####################################################
################### Cell 1 #########################
####################################################

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

####################################################
################### Cell 2 #########################
####################################################

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("/home/iblutman155201/dog-project/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

####################################################
################### Cell 3 #########################
####################################################

import cv2
import matplotlib.pyplot as plt
# % matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('/home/iblutman155201/dog-project/haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
# plt.show()

####################################################
################### Cell 4 #########################
####################################################

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

####################################################
############ Cell 5 - implementation ###############
####################################################

human_files_short = human_files[:100]
dog_files_short = train_files[:100]

hf_face_counter = 0
df_face_counter = 0

for human_file, dog_file in zip(human_files_short, dog_files_short):
    hf_face_counter += face_detector(human_file)
    df_face_counter += face_detector(dog_file)

print('%d percent of the first 100 images in human_files have a detected human face.' % hf_face_counter)
print('%d percent of the first 100 images in dog_files have a detected human face.' % df_face_counter)

####################################################
################### Cell 6 #########################
####################################################

from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

####################################################
################### Cell 7 #########################
####################################################

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

####################################################
################### Cell 8 #########################
####################################################

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

####################################################
################### Cell 9 #########################
####################################################

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

####################################################
################### Cell 10 ########################
####################################################
### Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
hf_dog_counter = 0
df_dog_counter = 0

for human_file, dog_file in zip(human_files_short, dog_files_short):
    hf_dog_counter += dog_detector(human_file)
    df_dog_counter += dog_detector(dog_file)

print('%d percent of the first 100 images in human_files have a detected dog.' % hf_dog_counter)
print('%d percent of the first 100 images in dog_files have a detected human dog.' % df_dog_counter)

####################################################
################### Cell 11 ########################
####################################################
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

####################################################
################### Cell 11 ########################
####################################################
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.
# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224,224, 3), name='block1_conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

# Classification block
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dense(133, activation='softmax', name='predictions'))

model.summary()
