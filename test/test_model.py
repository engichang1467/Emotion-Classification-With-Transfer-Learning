import numpy as np
import cv2
import pytest
from pathlib import Path
from your_model_module import (
    build_model,
)  # Adjust this import based on your actual code structure

# Assuming your model is defined in a module that's accessible,
# and the build_model function is correctly imported.

# Load model globally if it's not too resource-intensive or use a fixture to manage setup and teardown.
model = build_model(
    n_hidden=4, n_neurons=1000, learning_rate=0.007, activation="relu", dropout_rate=0.3
)

CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

# Update this to the actual path where your test images are located
TEST_IMAGE_DIR = Path.cwd().joinpath("A3-Data/Phoebe/images/unknown")


@pytest.fixture
def test_image():
    def _method(filename):
        image_path = TEST_IMAGE_DIR.joinpath(filename)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"No image found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        return image

    return _method


def classify_emotion(image, model):
    image_array = np.array([image])  # Model expects a batch of images
    output_tensor = model.predict(image_array)
    return output_tensor


@pytest.mark.parametrize(
    "image_filename",
    [
        "52_31.jpg",
        "41_06.jpg",
        "19_02.jpg",
        "11_01.jpg",
        "10_51.jpg",
        "9_41.jpg",
        "8_01.jpg",
        "1_01.jpg",
    ],
)
def test_classify_emotion(image_filename, test_image):
    image = test_image(image_filename)
    output_tensor = classify_emotion(image, model)
    output_list = output_tensor[0].tolist()

    assert isinstance(output_list, list), "Output should be a list"
    assert len(output_list) == len(
        CLASSES
    ), f"Output list should have {len(CLASSES)} elements"
    print(f"Tested {image_filename}: {output_list}")
