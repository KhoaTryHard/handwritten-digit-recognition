import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

from digit_pipeline.models import build_base_digit_cnn
from project_paths import project_file


BATCH_SIZE = 64
EPOCHS = 5
OUTPUT_MODEL = project_file("models", "stage_01_mnist_base.keras")


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def main() -> None:
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    model = build_base_digit_cnn()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_images, test_labels),
    )

    _, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    predictions = model.predict(test_images, verbose=0)

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Prediction for first test image: {np.argmax(predictions[0])}")

    model.save(OUTPUT_MODEL)
    print(f"Saved model to {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
