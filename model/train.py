from pathlib import Path
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
)


CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

# Define and move to dataset directory
datasetdir = Path.cwd().joinpath("A3-Data/Fer2013/images")
batch_size = 1


def generators(shape, preprocessing):
    """
    Create the training and validation datasets for a given image shape.
    """
    imgdatagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        horizontal_flip=True,  # data augmentation
        validation_split=0.2,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        datasetdir,
        target_size=(height, width),
        classes=CLASSES,
        batch_size=batch_size,
        subset="training",
        class_mode="sparse",
    )

    val_dataset = imgdatagen.flow_from_directory(
        datasetdir,
        target_size=(height, width),
        classes=CLASSES,
        batch_size=batch_size,
        subset="validation",
        class_mode="sparse",
    )

    return train_dataset, val_dataset


def build_model(
    n_hidden=1,
    n_neurons=100,
    learning_rate=0.005,
    activation="sigmoid",
    dropout_rate=0.0,
):
    """
    Use a pretrained model to train on X_data, y_data and return the tuned model.
    """
    # Initialize our VGG16 imagenet
    conv_model = keras.applications.vgg16.VGG16(
        weights="imagenet", include_top=False, input_shape=(48, 48, 3)
    )

    for layer in conv_model.layers:
        layer.trainable = False

    # Initialize our neural network
    model = Sequential()

    # Add our VGG16 into the network
    model.add(conv_model)
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(BatchNormalization())

    # Adding hidden layers
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, kernel_initializer="he_uniform"))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_neurons, kernel_initializer="he_uniform"))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    # Adding output layer
    model.add(Dense(len(CLASSES), activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )

    return model


# Use this block to train a single model with specified parameters
train_dataset, val_dataset = generators(
    (48, 48), preprocessing=keras.applications.vgg16.preprocess_input
)

model = build_model(
    n_hidden=4, n_neurons=1000, learning_rate=0.007, activation="relu", dropout_rate=0.3
)
# model = build_model(n_hidden=1, n_neurons=500, learning_rate=0.01, activation="relu", dropout_rate=0.6)
# parameter: weight (set during training)
# hyper parameter: number of hidden layers, (set b4 training)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    workers=-1,
    epochs=150,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)
