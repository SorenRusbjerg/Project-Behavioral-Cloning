from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras import layers
from keras.models import Model, load_model
import tensorflow as tf
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn import utils
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import cv2
from pathlib import Path

loadModel = 'load_train'


# Reset Keras Session
def reset_keras():
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    # print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))

# Clear cuda memory
# K.clear_session()
reset_keras()

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
nn_Imgsize = 128

def ResizeNormalizeLambda(image, imgsize=224):
    import tensorflow as tf
    image = tf.image.resize_images(image, (imgsize, imgsize))
    image = (tf.to_float(image)-127.0)/256.0
    return image

# Get pretrained model from MobileNetV2 with no fully connected layers in the top, and avg pooling in the final layer (alpha=1.0 will use default network size)
model_pretrained = MobileNet(input_shape=(nn_Imgsize, nn_Imgsize, 3),
                             alpha=1.0, include_top=False, weights='imagenet')

#model_pretrained = InceptionV3(weights='imagenet', include_top=False,
#                        input_shape=(nn_Imgsize,nn_Imgsize,3))

# Print model layers
#model_pretrained.summary()
# Freeze layers
freezeLayers(model_pretrained, 'conv2d_94')

# Input layer
model_input = layers.Input(shape=imgShape)

# Crop input images
crop_out = layers.Cropping2D(cropping=((cropTop, cropBottom), (0, 0)))(model_input)

# Re-sizes and Normalize the input with Kera's Lambda layer & attach to model_input
resized_input = layers.Lambda(ResizeNormalizeLambda, arguments={'imgsize': nn_Imgsize})(crop_out)

# Connect layers
# deep_nn_out = model_pretrained(resized_input)

conv = layers.Conv2D(8, (5,5), padding='same', activation='relu')(resized_input)
conv = layers.MaxPooling2D((2,2),padding='same')(conv)
#conv = layers.BatchNormalization()(conv)
conv = layers.Conv2D(32, (5,5), padding='same', activation='relu')(conv)
conv = layers.MaxPooling2D((4,4),padding='same')(conv)
conv = layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv)
conv = layers.AveragePooling2D((4,4),padding='same')(conv)

conv_out = layers.Flatten()(conv)
x = layers.Dropout(0.5)(conv_out)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(12, activation='relu')(x)
x = layers.Dropout(0.5)(x)
anglePredict = layers.Dense(1)(x)

# Create final model
model = Model(inputs=model_input, outputs=anglePredict)
model_cropInput = Model(inputs=model_input, outputs=crop_out)

# setting up optimizer
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.summary()


## Get data images and steering angles
sideOffset = 0.2
batchSize = 16
csvFolder = 'Training_Turn'
csvFile = csvFolder+'/driving_log.csv'
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
                dataPath = Path(line[0])
                dataPath = Path(csvFolder).joinpath(dataPath.parts[-2])
                imgCenter = dataPath.joinpath(Path(line[0]).name)
                imgLeft = dataPath.joinpath(Path(line[1]).name)
                imgRight = dataPath.joinpath(Path(line[2]).name)
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
if loadModel == 'load_train':
    model = load_model('model.h5', custom_objects={'tf': tf, 'ResizeNormalizeLambda': ResizeNormalizeLambda})
    model_cropInput = load_model('model_cropinput.h5')

if loadModel == 'train' or 'load_train':
    fit_history = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples))/batchSize,
                                    validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batchSize), epochs = epochs, verbose = 1)
    model.save('model.h5')
    model_cropInput.save('model_cropinput.h5')
    # Plot training loss
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('mse loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training set','validation set'], loc='upper right')
    plt.show()
elif loadModel == 'load':
    model = load_model('model.h5', custom_objects={'tf': tf, 'ResizeNormalizeLambda': ResizeNormalizeLambda})
    model_cropInput = load_model('model_cropinput.h5')
else:
    print('unknown loadmodel ', loadModel)

# Plot Cropped output
# model.get_output_at()
xval, yval = next(train_generator)

imgCropped = model_cropInput.predict(xval)
print(imgCropped[0,0:5,0:5,0])

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for n, ax in enumerate(axs):
    ax[0].imshow(xval[n])
    ax[1].imshow(cv2.cvtColor(imgCropped[n], cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

# Plot single estimations
steer = model.predict_on_batch(xval)
print()
print(yval)
print()
print(steer)

# Plot histogram
