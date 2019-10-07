from keras.applications.mobilenet import MobileNet
from keras import layers
from keras.models import Model
import tensorflow as tf
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn import utils
import csv
import numpy as np
import matplotlib.pyplot as plt

def freezeLayers(nn_model, layerName):
    # Freeze layers incl. and after layerName
    train = False
    for layer in nn_model.layers:
        if layer.name == layerName:
            train = True
            print()
            print('Start train!')
        layer.trainable = train
        print('layername: {}, train: {}'.format(layer.name, layer.trainable))


imgShape = (160, 320, 3)
cropTop = 62
cropBottom = 20
nn_Imgsize = 224

# Get pretrained model from MobileNetV2 with no fully connected layers in the top, and avg pooling in the final layer (alpha=1.0 will use default network size)
model_pretrained = MobileNet(input_shape=(nn_Imgsize, nn_Imgsize, 3),
                             alpha=1.0, include_top=False, weights='imagenet', pooling='avg')
# Print model layers
model_pretrained.summary()
# Freeze layers
freezeLayers(model_pretrained, 'conv_pw_13')

# Input layer
model_input = layers.Input(shape=imgShape)

# Crop input images
crop_out = layers.Cropping2D(cropping=((cropTop, cropBottom), (0, 0)))(model_input)

# Re-sizes the input with Kera's Lambda layer & attach to model_input
resized_input = layers.Lambda(lambda image: tf.image.resize_images(image, (nn_Imgsize, nn_Imgsize)))(crop_out)

# Normalize inputs
normalized_input = layers.Lambda(lambda image: (image-127.0)/128)(resized_input)

# Connect layers
deep_nn_out = model_pretrained(normalized_input)
x = layers.Dropout(0.5)(deep_nn_out)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
anglePredict = layers.Dense(1)(x)

# Create final model
model = Model(inputs=model_input, outputs=anglePredict)


# setting up optimizer
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
model.summary()


## Get data images and steering angles
sideOffset = 0.1
batchSize = 64
csvFile = 'TrainingData/driving_log.csv'
# Load csv data
lines = []
with open(csvFile) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Create generator for getting batch data
def DataGenerator(samples, batchSize=64):
    N_samples = len(samples)

    while True:
        utils.shuffle(samples)

        for offset in range(0, N_samples, batchSize):
            batch = samples[offset:offset+batchSize]
            
            images = []
            measurements = []
            for line in batch:
                # Insert images center, left, right
                imgCenter = line[0]
                imgLeft = line[1]
                imgRight = line[2]    
                images.append(ndimage.imread(imgCenter))
                images.append(ndimage.imread(imgLeft))
                images.append(ndimage.imread(imgRight))

                # Insert steer angle
                steerAngle = float(line[3])
                measurements.append(steerAngle)
                measurements.append(steerAngle + sideOffset)
                measurements.append(steerAngle - sideOffset)

            X_samples = np.array(images)
            y_samples = np.array(measurements)
            yield X_samples, y_samples


train_generator = DataGenerator(train_samples, batchSize=batchSize)
validation_generator = DataGenerator(validation_samples, batchSize=batchSize)

## Train model
epochs = 5

fit_history = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples))/batchSize,
                                  validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batchSize), epochs = epochs, verbose = 1)


model.save('model.h5')

# Plot training loss
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('mse loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'], loc='upper right')
plt.show()
