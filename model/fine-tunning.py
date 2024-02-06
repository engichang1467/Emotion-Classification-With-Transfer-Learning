import numpy as np
from tensorflow.keras import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from train import build_model, generators

keras_classifier = KerasClassifier(
    model=build_model,
    n_hidden=1,
    n_neurons=1000,
    learning_rate=0.007,
    activation="relu",
    dropout_rate=0.05,
)

# dataset, _ = generators((331, 331), "nasnet", validation_split=0.2)
dataset, _ = generators((224, 224), "vgg", validation_split=0.2)

X_data, y_data, batch_index = [], [], 0

while batch_index <= dataset.batch_index:
    data = dataset.next()
    print(data[0])
    print(data[1])
    X_data.append(data[0])
    y_data.append(data[1])
    batch_index += 1

X_data, y_data = np.asarray(X_data), np.asarray(y_data)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_data, y_data, test_size=0.2, random_state=1
)

X_valid, X_valid = X_train.squeeze(), X_valid.squeeze()

param_distribs = {
    "n_hidden": (1, 2, 3, 4),
    "n_neurons": (100, 200, 300, 400, 500, 1000),
    "learning_rate": (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01),
    "activation": ("sigmoid", "relu"),
    "dropout_rate": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
}

rnd_search_cv = RandomizedSearchCV(keras_classifier, param_distribs, n_iter=40, cv=3)
rnd_search_cv.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=15)],
)

print(rnd_search_cv.best_params_)
